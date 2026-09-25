r"""Selektiver State-Space-Kern (Mamba, Gu & Dao 2023, arXiv:2312.00752) -- der ``mamba``-Slot.

Warum reines PyTorch und nicht ``mamba-ssm``
---------------------------------------------
Geprüft (Schritt 0 dieses Arbeitsstrangs): ``pip install mamba-ssm`` scheitert auf dieser
CPU-only-Windows-Umgebung an einer harten Abhängigkeit von ``nvcc`` (CUDA-Compiler) --
das offizielle Paket baut eigene CUDA-Kernel für den parallelen Scan und lässt sich ohne
CUDA-Toolkit nicht einmal installieren (nicht nur nicht ausführen). Diese Datei
implementiert stattdessen die Rekurrenz selbst, sequenziell, ohne eigene CUDA-Kernel --
siehe Modul-Docstring-Abschnitt "Vereinfachungen" unten für das, was dabei bewusst
wegfällt.

Nur die Zelle, kein Block -- keine Fehlerquelle mehr, als bei xLSTM
--------------------------------------------------------------------
Genau wie ``nn.GRU``/``nn.LSTM``/``XLSTMCore`` (:mod:`twinlink_ml.models.xlstm`) ist
:class:`MambaCore` (Fabrik :func:`build_mamba_core_class`) eine rohe rekurrente Zelle mit
demselben ``forward(sequence, hx=None) -> (output, hidden)``-Aufrufvertrag, kein
Encoder-Decoder-Paar. Anders als beim Transformer-Kern (der einen eigenen Zweig
``_build_transformer_network`` brauchte, weil ``nn.TransformerEncoder``/``-Decoder`` das
``(input, hidden)``-Aufrufmuster nicht kennen) reicht bei Mamba **eine zusätzliche Zeile**
im ``cell_type``-Dict von ``torch_sequence._build_network`` -- der Trainer
(``TorchSequenceModel.fit``/``predict``/``state_dict``/``load_checkpoint``) bleibt
unverändert, wie bei xLSTM.

Die Rekurrenz, Schritt für Schritt (das ist der ganze Grund für diese Architektur)
------------------------------------------------------------------------------------
Eingabe zum Zeitschritt :math:`t`: :math:`u_t \in \mathbb{R}^{d_{model}}`, zuerst
``RMSNorm``-normalisiert (Referenz: ``Block.forward`` in ``state-spaces/mamba`` folgt
strikt ``residual = hidden + residual; hidden = norm(residual); hidden = mixer(hidden)``
-- der Mixer bekommt nie den unnormierten Residualstrom zu sehen). Ohne diese Norm kann
sich der autoregressive Rollout dieses Projekts (``last = last + head(...)`` in
``torch_sequence.Network.forward``) aufschaukeln: ein einmal etwas zu großer Ausgang wird
im nächsten Schritt zu einem noch größeren Eingang, über 36 Horizontschritte exponentiell
-- diagnostiziert an einem echten Trainingslauf (Schritt-2-Vergleich, ``n_train=16``,
Teilnetz ``energy_target``, Epoche 12: der rohe Layer-Eingang wuchs innerhalb von 5
Schritten von ~2 auf >1e19, `h` kollabierte zu NaN). GRU/LSTM sind dagegen strukturell
robust (sättigende tanh/sigmoid-Gates), ``nn.Transformer{Encoder,Decoder}Layer`` bringen
LayerNorm eingebaut mit -- diese selbst geschriebene SSM-Zelle brauchte die Norm
nachträglich, dann projiziert über ``in_proj`` in zwei Hälften :math:`x_t, z_t \in
\mathbb{R}^{d_{inner}}` (``d_inner = expand * d_model``, ``expand = 2``).

**Gelernte, zeitunabhängige Parameter:**

.. math:: A = -\exp(A_{log}) \in \mathbb{R}^{d_{inner} \times d_{state}}
   \qquad D \in \mathbb{R}^{d_{inner}}

``A_log`` initialisiert mit :math:`\log(n)`, :math:`n = 1 \ldots d_{state}`, je Zeile
identisch wiederholt -- die S4D-real-Initialisierung aus der Referenzimplementierung
(garantiert :math:`A < 0`, also Stabilität, unabhängig vom Trainingsfortschritt).

**Selektive, eingabeabhängige Parameter** (linear aus :math:`x_t`, NACH SiLU-Aktivierung
-- das macht die Auswahl "selektiv" statt wie bei S4 zeitinvariant):

.. math:: \Delta_t = \operatorname{softplus}(W_\Delta x_t + b_\Delta) \in \mathbb{R}_+^{d_{inner}}
   \qquad B_t = W_B x_t \in \mathbb{R}^{d_{state}}
   \qquad C_t = W_C x_t \in \mathbb{R}^{d_{state}}

(``W_\Delta`` läuft über eine Tiefpass-Projektion mit Rang ``dt_rank = max(1, d_model //
16)`` -- Parametersparsamkeit aus der Referenz, bei diesem Projekts kleinem ``d_model``
kaum relevant, aber übernommen für Werktreue.)

**Diskretisierung** (Euler-Näherung für :math:`\bar B`, wie in der Referenz):

.. math:: \bar A_t[i,n] = \exp(\Delta_t[i] \cdot A[i,n]) \qquad
   \bar B_t[i,n] = \Delta_t[i] \cdot B_t[n]

**Zustandsupdate -- O(1) pro Schritt, das eigentliche Ziel dieser Architektur:**

.. math:: h_t[i,n] = \bar A_t[i,n] \cdot h_{t-1}[i,n] + \bar B_t[i,n] \cdot x_t[i]

:math:`h_t \in \mathbb{R}^{d_{inner} \times d_{state}}` hängt ausschließlich von
:math:`h_{t-1}` und den Größen des aktuellen Schritts ab -- keine Referenz auf
:math:`t' < t-1`, keine wachsende Sequenz wird bei jedem Schritt neu durchlaufen (anders
als beim Transformer-Decoder, der bei Schritt :math:`t` über alle :math:`t` bisherigen
Token neu attendiert, siehe ``torch_sequence._build_transformer_network``-Docstring und
Befund 37, ``docs/04_stand_ap1.md``). Exakt dieselbe Eigenschaft, die eine
GRU/LSTM-Zelle hat: ein fortgeschriebener Zustand fester Größe, keine wachsende
Kontextmenge.

**Ausgabe:**

.. math:: y_t[i] = \sum_n C_t[n] \cdot h_t[i,n] + D[i] \cdot x_t[i]
   \qquad y_t \leftarrow y_t \odot \operatorname{SiLU}(z_t)
   \qquad \text{out}_t = W_{out}\, y_t \in \mathbb{R}^{d_{model}}

Für eine volle Sequenz (z. B. die Historie) wird dieselbe Rekurrenz in einer Python-
``for``-Schleife sequenziell über die Zeit ausgeführt -- kein paralleler (Cumsum-/Scan-)
Trick, wie ausdrücklich beauftragt (keine eigene CUDA-Kernel-Entwicklung, kein Versuch,
die parallele Scan-Performance der Referenz nachzubauen).

Vereinfachungen und Annahmen, offen aufgelistet (analog zu ``xlstm.py``)
--------------------------------------------------------------------------
1. **Kein kausaler Depthwise-Conv1d vor der SSM.** Im offiziellen Mamba-Block sitzt vor
   der Selektionsprojektion ein kurzer (Kernel 4) kausaler Conv1d für lokalen Kontext.
   Hier weggelassen -- bestätigt in der Schritt-0-Rückmeldung als kein Blocker für die
   O(1)-Kerneigenschaft, bei Bedarf nachrüstbar (bräuchte einen kleinen FIFO-Puffer als
   Teil von ``hidden``, bliebe O(1) pro Schritt).
2. **``dt_proj``-Bias mit ``nn.Linear``-Standardinitialisierung**, nicht mit der
   speziellen ``dt_init``-Softplus-Umkehr-Initialisierung der Referenz (steuert dort den
   anfänglichen effektiven Zeitschritt). Trainingsdynamik-, keine Korrektheitsfrage.
3. **Nur die sequenzielle Rekurrenz**, nicht der parallele assoziative Scan der
   Referenz -- bei den Sequenzlängen dieses Projekts (Historie 16-32, Horizont bis 36)
   ein Geschwindigkeits-, kein Korrektheitskompromiss (dieselbe Begründung wie bei
   xLSTM, Punkt 6 dort).
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_mamba_core_class"]

def build_mamba_core_class(torch_module: Any, state_size: int, expand: int = 2) -> type:
    """Erzeugt die ``MambaCore``-Klasse zur Laufzeit, mit ``torch_module`` als Basis.

    ``state_size``/``expand`` sind hier gebunden (nicht Teil des Konstruktoraufrufs) --
    damit hat ``MambaCore.__init__`` dieselbe Signatur wie ``nn.GRU``/``nn.LSTM``/
    ``XLSTMCore`` (``input_size, hidden_size, num_layers, dropout, batch_first``) und passt
    unverändert in das ``cell_type``-Dict von ``torch_sequence._build_network``. Dieselbe
    Fabrik-statt-Modulklasse-Begründung wie bei ``build_xlstm_core_class``: ``torch.nn.Module``
    muss zur Definitionszeit verfügbar sein, das Paket muss ohne PyTorch importierbar bleiben.
    """
    nn_module = torch_module.nn
    functional = nn_module.functional
    base: type = nn_module.Module

    class _MambaLayer(base):  # type: ignore[misc]
        """Eine selektive SSM-Schicht: Zustand ``h`` -> Zustand, ein Zeitschritt je Aufruf."""

        def __init__(self, input_size: int, hidden_size: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.d_inner = expand * hidden_size
            self.dt_rank = max(1, hidden_size // 16)

            # Referenz (state-spaces/mamba, `Block.forward`): "Add -> LN -> Mixer" --
            # der Mixer bekommt NIE den unnormierten Residualstrom zu sehen, nur die
            # normalisierte Ansicht davon. Hier fehlte diese Norm zunaechst komplett;
            # ohne sie kann der autoregressive Rollout (`last = last + head(...)` in
            # `torch_sequence.Network.forward`) sich aufschaukeln: ein einmal etwas zu
            # grosser Ausgang wird im naechsten Schritt zu einem noch groesseren Eingang,
            # ueber 36 Horizontschritte hinweg exponentiell (diagnostiziert an einem
            # echten Trainingslauf, n_train=16, Teilnetz energy_target, Epoche 12: der
            # rohe Layer-Eingang selbst wuchs innerhalb von 5 Schritten von ~2 auf >1e19,
            # bevor der Zustand nach NaN kollabierte -- ein reiner Delta-Clamp half nicht,
            # weil `b_t`/`c_t` aus derselben unnormierten Eingabe ebenso explodierten).
            # GRU/LSTM sind dagegen strukturell robust (saettigende tanh/sigmoid-Gates),
            # `nn.Transformer{Encoder,Decoder}Layer` bringen LayerNorm eingebaut mit --
            # die selbst geschriebene SSM-Zelle hatte keine Entsprechung.
            self.norm = nn_module.RMSNorm(input_size)
            self.in_proj = nn_module.Linear(input_size, 2 * self.d_inner, bias=False)
            self.x_proj = nn_module.Linear(self.d_inner, self.dt_rank + 2 * state_size, bias=False)
            self.dt_proj = nn_module.Linear(self.dt_rank, self.d_inner, bias=True)

            # S4D-real-Initialisierung (Referenz): A = -(1, 2, ..., state_size), je
            # d_inner-Zeile identisch -- garantiert A < 0 unabhaengig vom Training.
            log_a_init = torch_module.log(
                torch_module.arange(1, state_size + 1, dtype=torch_module.float32)
            )
            self.A_log = nn_module.Parameter(log_a_init.unsqueeze(0).repeat(self.d_inner, 1))
            self.D = nn_module.Parameter(torch_module.ones(self.d_inner))
            # Referenzkonvention (state-spaces/mamba, mamba_simple.py): A_log/D tragen
            # die Zeitskalen- bzw. Skip-Information der SSM direkt (keine gelernten
            # Gewichte im ueblichen Sinn) -- Weight Decay wuerde die S4D-real-Vielfalt
            # der Zerfallsraten ueber viele Epochen Richtung A_log=0 homogenisieren.
            # `TorchSequenceModel._param_groups` (torch_sequence.py) liest genau dieses
            # Attribut generisch, kein Mamba-Sonderfall dort.
            self.A_log._no_weight_decay = True
            self.D._no_weight_decay = True
            self.out_proj = nn_module.Linear(self.d_inner, hidden_size, bias=False)

        def init_state(self, batch_size: int, reference: Any) -> Any:
            """Nullzustand, Form/Geraet/Dtype von ``reference`` (Eingabetensor) uebernommen."""
            return reference.new_zeros(batch_size, self.d_inner, state_size)

        def step(self, x_t: Any, h_prev: Any) -> tuple[Any, Any]:
            """Ein Zeitschritt -- O(1): haengt nur von ``h_prev`` und ``x_t`` ab.

            Gleichungen siehe Moduldocstring. ``self.norm`` zuerst, wie in der Referenz
            (siehe ``__init__``-Kommentar) -- alles Folgende sieht nur die normalisierte
            Eingabe, nie den unbeschraenkten Rohwert von ``x_t`` selbst.
            """
            xz = self.in_proj(self.norm(x_t))
            x, z = xz.chunk(2, dim=-1)
            x = functional.silu(x)

            projected = self.x_proj(x)
            delta_in, b_t, c_t = torch_module.split(
                projected, [self.dt_rank, state_size, state_size], dim=-1
            )
            delta = functional.softplus(self.dt_proj(delta_in))
            a = -torch_module.exp(self.A_log)

            discrete_a = torch_module.exp(delta.unsqueeze(-1) * a.unsqueeze(0))
            discrete_b = delta.unsqueeze(-1) * b_t.unsqueeze(1)
            h = discrete_a * h_prev + discrete_b * x.unsqueeze(-1)

            y = (h * c_t.unsqueeze(1)).sum(dim=-1) + self.D.unsqueeze(0) * x
            y = y * functional.silu(z)
            return self.out_proj(y), h

    class MambaCore(base):  # type: ignore[misc]
        """Selektiver-SSM-Stapel mit demselben Aufrufvertrag wie ``nn.GRU``/``nn.LSTM``.

        ``forward(input, hx=None) -> (output, hidden)``, ``batch_first=True`` erzwungen
        (die einzige Aufrufform, die ``torch_sequence.py`` je verwendet). ``hidden`` ist
        ein Tupel von ``num_layers`` Zustandstensoren -- wie bei GRU/LSTM/xLSTM wird es
        von ``torch_sequence.py`` nirgends ausgepackt, sondern nur durchgereicht.
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            dropout: float = 0.0,
            batch_first: bool = True,
        ) -> None:
            super().__init__()
            if not batch_first:
                raise ValueError("MambaCore unterstuetzt nur batch_first=True")
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.layers = nn_module.ModuleList(
                _MambaLayer(input_size if index == 0 else hidden_size, hidden_size)
                for index in range(num_layers)
            )
            self.dropout = nn_module.Dropout(dropout)

        def forward(self, sequence: Any, hx: Any | None = None) -> tuple[Any, Any]:
            """Historie oder Einzelschritt kodieren -- Sequenzlaenge ist beliebig.

            Sequenziell ueber die Zeit (kein paralleler Scan), siehe Moduldocstring.
            """
            batch_size, seq_len, _ = sequence.shape
            if hx is None:
                states = [layer.init_state(batch_size, sequence) for layer in self.layers]
            else:
                states = list(hx)

            layer_input = sequence
            for layer_index, layer in enumerate(self.layers):
                state = states[layer_index]
                outputs = []
                for step in range(seq_len):
                    out_t, state = layer.step(layer_input[:, step, :], state)
                    outputs.append(out_t)
                states[layer_index] = state
                layer_input = torch_module.stack(outputs, dim=1)
                if layer_index < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            return layer_input, tuple(states)

    return MambaCore
