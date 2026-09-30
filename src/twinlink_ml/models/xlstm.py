r"""mLSTM-Kern nach Beck et al. (2024), arXiv:2405.04517 -- der ``xlstm``-Slot.

**Nur der mLSTM-Block, kein sLSTM.** In der Notation des Papers ist das xLSTM[1:0]
(ausschließlich mLSTM-Zweige, keiner der skalar-speichrigen sLSTM-Zweige). Der
Arbeitsplan (``05_arbeitsplan_ap1.md``) gibt vor: "erst mLSTM-lastig, sLSTM nur bei
Bedarf". Wer unter dem Label ``"xlstm"`` einen sLSTM-Anteil erwartet, findet hier
keinen -- das ist Absicht, nicht eine Lücke, die noch zu schließen wäre.

**Nur die Zelle, kein Block.** Genau wie ``nn.GRU``/``nn.LSTM`` in
:mod:`twinlink_ml.models.torch_sequence` rohe rekurrente Zellen sind (kein
Encoder-Decoder, kein Residual-Backbone), implementiert :class:`XLSTMCore` (Fabrik
:func:`build_xlstm_core_class`) nur die mLSTM-**Rekurrenz** -- Zellzustand, Normalisierer,
Gates. Was im offiziellen xLSTM-Block zusätzlich existiert (Hoch-/Herunterprojektion,
kausale Kurzfaltung vor Query/Key, mehrere Köpfe, Block-Residual mit Pre-LayerNorm) ist
hier **nicht** enthalten. Das ist konsistent mit der bestehenden Abstraktionsebene dieses
Moduls (``core=`` ist ein rekurrenter Kern, kein Block) und keine verkürzte Kopie eines
Blocks, dessen Teile fehlen.

Woher die Gleichungen stammen und wie sicher das ist
------------------------------------------------------
Das Paper selbst war über die verfügbaren Werkzeuge dieser Sitzung nicht mit vollem
Gleichungssatz abrufbar (die HTML-Fassung auf arXiv bricht an der mLSTM-Stelle ab). Die
hier implementierten Gleichungen sind stattdessen gegen die **offizielle Referenz-
implementierung** geprüft (``NX-AI/xlstm``, Dateien ``xlstm/blocks/mlstm/cell.py`` und
``xlstm/blocks/mlstm/backends.py``, Funktion ``recurrent_step_stabilized_simple``,
abgerufen über den Quelltext auf GitHub). Das ist eine andere Art von Quelle als das
Paper selbst -- Code statt Gleichungsnummern --, aber eine, die tatsächlich lauffähig
und von den Originalautoren gepflegt ist. Übernommen aus dieser Referenz:

* Die Rekurrenz für Zellzustand ``C_t``, Normalisierer ``n_t`` und Max-Stabilisierer
  ``m_t`` (siehe :meth:`_XLSTMLayer.step`), einschließlich der Reihenfolge im
  Außenprodukt (``k`` zuerst, ``v`` zweitens -- das legt fest, welcher Index beim
  Auslesen mit ``q`` kontrahiert wird).
* Die Skalierung von ``k`` mit :math:`1/\sqrt{d}` (nicht von ``q``).
* Die Gatter-Voraktivierungen aus ``concat([q, k, v])`` statt direkt aus ``x_t`` --
  ein Detail, das in der Zellklasse selbst steht (``self.igate``/``self.fgate`` mit
  ``in_features = 3 * embedding_dim``), nicht in den Blockdiagrammen des Papers.
* Der Nenner ``max(|n_t^T q_t|, exp(-m_t)) + eps`` statt des im Fließtext oft
  vereinfachten ``max(|n_t^T q_t|, 1)`` -- die ``exp(-m_t)``-Variante ist die im
  Log-Raum stabilisierte, äquivalente Form (siehe Herleitung unten).

**Geklärt (echt gegen die Referenz geprüft, ``NX-AI/xlstm``, ``mlstm_cell.py``):** die
abschließende Normalisierung (``outnorm``) wird **vor** der Multiplikation mit dem
Output-Gate ``o_t`` angewendet -- exakt wie hier implementiert (auf :math:`\tilde h_t`,
siehe unten). Ursprünglich als unsicher markiert (nur der Quelltextausschnitt zur
Rekurrenz selbst war zuvor abgerufen worden, nicht der der Zellklasse); die spätere
Prüfung des tatsächlichen Aufrufs bestätigt diese Reihenfolge.

Gleichungen, wie hier implementiert
------------------------------------
Pro Zeitschritt :math:`t`, Zustand :math:`(C_{t-1}, n_{t-1}, m_{t-1})`:

.. math::
   q_t &= W_q x_t + b_q \qquad k_t = \frac{W_k x_t + b_k}{\sqrt{d}}
   \qquad v_t = W_v x_t + b_v

   \tilde\imath_t, \tilde f_t &= W_{if}\,[q_t; k_t; v_t] + b_{if}
   \qquad \log f_t = -\operatorname{softplus}(-\tilde f_t)

   m_t &= \max(\log f_t + m_{t-1},\ \tilde\imath_t)

   i'_t &= \exp(\tilde\imath_t - m_t) \qquad f'_t = \exp(\log f_t + m_{t-1} - m_t)

   C_t &= f'_t\,C_{t-1} + i'_t\,(k_t \otimes v_t) \qquad n_t = f'_t\,n_{t-1} + i'_t\,k_t

   \tilde h_t &= \operatorname{LayerNorm}\!\left(\frac{q_t^\top C_t}
   {\max(|q_t^\top n_t|,\ \exp(-m_t)) + \varepsilon}\right)

   o_t &= \sigma(W_o x_t + b_o) \qquad h_t = o_t \odot \tilde h_t

Warum diese Rekurrenz nicht überläuft (die geforderte Stabilisierung)
------------------------------------------------------------------------
Ohne :math:`m_t` wären :math:`i_t = \exp(\tilde\imath_t)` und :math:`f_t = \exp(\log f_t)`
unbeschränkt bzw. akkumulieren sich über viele Schritte multiplikativ zu Werten, die
``float32`` überlaufen (das ist genau der Fall, gegen den die Aufgabenstellung warnt).
:math:`m_t` ist als laufendes Maximum der log-Gatter-Summen definiert
(:math:`\max(\log f_t + m_{t-1}, \tilde\imath_t)` -- dieselbe Log-Sum-Exp-Idee wie bei
einer stabilen Softmax). Dadurch sind beide Exponenten in :math:`i'_t, f'_t` konstruktions-
bedingt :math:`\le 0`, also :math:`i'_t, f'_t \in (0, 1]` -- **beschränkt für jede
Eingabe**, unabhängig von Sequenzlänge oder Gewichtsinitialisierung. Das ist keine
Näherung des exakten Werts, sondern eine mathematisch äquivalente Umformulierung (siehe
Ableitung in den Kommentaren von ``backends.py`` der Referenz): der Nenner
``max(|n_t^T q_t|, exp(-m_t))`` kompensiert exakt den Faktor, um den ``C_t``/``n_t``
gegenüber ihren unstabilisierten Gegenstücken re-skaliert sind.

Vereinfachungen und Annahmen, offen aufgelistet
--------------------------------------------------
1. **Ein Kopf** (``NH = 1``), nicht mehrere. Der offizielle Block splittet ``q``/``k``/``v``
   in mehrere Köpfe mit kleinerer Kopf-Dimension. Bei den hier üblichen ``hidden_size``
   (8-64) und der kleinen Eingangsdimension dieses Projekts (~8 Kanäle je Teilmodell)
   hätte ein zusätzlicher ``num_heads``-Hyperparameter in ``TorchSequenceConfig``
   eingeführt werden müssen, den die harte Integrationsvorgabe (Aufrufvertrag von
   ``nn.GRU``/``nn.LSTM``, keine neuen Konstruktorparameter) nicht vorsieht. Ein Kopf ist
   der Grenzfall ``NH=1`` derselben Gleichungen, keine andere Architektur.
2. **Kein Up-/Down-Projektion, keine kausale Kurzfaltung vor q/k.** Siehe Moduldocstring
   oben -- das ist die Zelle, nicht der Block.
3. **Forget-Gate-Bias auf +3 initialisiert**, ein Sonderfall des verbreiteten LSTM-Tricks
   (Jozefowicz et al. 2015), der frühes Vergessen erschwert. Das ist **keine** dem Paper
   entnommene Vorgabe -- ich konnte über die verfügbaren Quellen keine explizite
   Initialisierungsvorschrift der Referenz für dieses Gate finden. Neutral wäre
   ``bias=0``; +3 ist eine Trainingserleichterung, deren Wirkung ungeprüft ist.
4. **``bias=True`` auf allen linearen Projektionen** (``q``, ``k``, ``v``, ``o``, Gatter).
   **Bewusste Abweichung von der Referenz** (echt geprüft, ``NX-AI/xlstm``,
   ``mlstm_layer.py``): dort ist der Konfigurations-Standard ``bias=False`` für die
   ``q``/``k``/``v``/``o``-Projektionen (``LinearHeadwiseExpand``/``proj_down``). Diese
   Implementierung übernimmt stattdessen die ``nn.Linear``-Voreinstellung (``bias=True``)
   -- kein Korrektheitsfehler (zusätzlicher Freiheitsgrad, kein struktureller Unterschied),
   aber eine Abweichung vom Referenz-Standardwert, nicht nur eine ungeprüfte Annahme.
5. **``eps = 1e-6``** im Nenner, ein Wert, keine aus dem Paper übernommene Zahl (die
   Referenz nennt ein ``eps``, aber der abgerufene Codeausschnitt zeigte keinen
   konkreten Zahlenwert).
6. **Nur die sequenzielle (rekurrente) Form implementiert**, nicht die parallele
   Matrixform, die die Referenz zusätzlich für schnelleres Training anbietet. Bei den
   Sequenzlängen dieses Projekts (Historie 16-32, Horizont bis 24) ist das ein
   Geschwindigkeits-, kein Korrektheitskompromiss -- explizit in Kauf genommen, um eine
   einzige, unzweideutige Implementierung der Rekurrenz zu haben statt zwei, die
   gegeneinander getestet werden müssten.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["build_xlstm_core_class"]


def build_xlstm_core_class(torch_module: Any) -> type:
    """Erzeugt die ``XLSTMCore``-Klasse zur Laufzeit, mit ``torch_module`` als Basis.

    Dieselbe Fabrik-statt-Modulklasse-Begründung wie bei ``_build_network`` in
    ``torch_sequence.py``: ``torch.nn.Module`` muss zur Definitionszeit verfügbar sein,
    und dieses Paket muss ohne PyTorch importierbar bleiben. Der Aufrufer (``_build_network``)
    importiert dieses Modul deshalb erst, nachdem ``_require_torch()`` bereits gelaufen ist.
    """
    nn_module = torch_module.nn
    functional = nn_module.functional
    base: type = nn_module.Module

    class _XLSTMLayer(base):  # type: ignore[misc]
        """Eine mLSTM-Schicht: Zustand ``(C, n, m)`` -> Zustand, ein Zeitschritt je Aufruf."""

        def __init__(self, input_size: int, hidden_size: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.w_q = nn_module.Linear(input_size, hidden_size)
            self.w_k = nn_module.Linear(input_size, hidden_size)
            self.w_v = nn_module.Linear(input_size, hidden_size)
            self.w_o = nn_module.Linear(input_size, hidden_size)
            self.w_i = nn_module.Linear(3 * hidden_size, 1)
            self.w_f = nn_module.Linear(3 * hidden_size, 1)
            # Forget-Gate-Bias-Trick (Jozefowicz et al. 2015), siehe Moduldocstring
            # Punkt 3 -- keine Paper-Vorgabe, eine übliche Trainingserleichterung.
            nn_module.init.constant_(self.w_f.bias, 3.0)
            self.norm = nn_module.LayerNorm(hidden_size)

        def init_state(self, batch_size: int, reference: Any) -> tuple[Any, Any, Any]:
            """Nullzustand, Form/Gerät/Dtype von ``reference`` (Eingabetensor) übernommen."""
            cell = reference.new_zeros(batch_size, self.hidden_size, self.hidden_size)
            normalizer = reference.new_zeros(batch_size, self.hidden_size)
            stabilizer = reference.new_zeros(batch_size, 1)
            return cell, normalizer, stabilizer

        def step(self, x_t: Any, state: tuple[Any, Any, Any]) -> tuple[Any, tuple[Any, Any, Any]]:
            """Ein Zeitschritt. Gleichungen siehe Moduldocstring."""
            cell, normalizer, stabilizer = state
            query = self.w_q(x_t)
            key = self.w_k(x_t) / math.sqrt(self.hidden_size)
            value = self.w_v(x_t)

            gate_input = torch_module.cat([query, key, value], dim=-1)
            i_preact = self.w_i(gate_input)
            f_preact = self.w_f(gate_input)
            log_f = -functional.softplus(-f_preact)

            m_new = torch_module.maximum(log_f + stabilizer, i_preact)
            i_act = torch_module.exp(i_preact - m_new)
            f_act = torch_module.exp(log_f + stabilizer - m_new)

            outer = torch_module.einsum("bi,bj->bij", key, value)
            cell_new = f_act.unsqueeze(-1) * cell + i_act.unsqueeze(-1) * outer
            normalizer_new = f_act * normalizer + i_act * key

            h_num = torch_module.einsum("bi,bij->bj", query, cell_new)
            qn_dot = torch_module.einsum("bi,bi->b", query, normalizer_new).unsqueeze(-1)
            denom = torch_module.maximum(qn_dot.abs(), torch_module.exp(-m_new)) + 1e-6
            h_tilde = self.norm(h_num / denom)

            output_gate = torch_module.sigmoid(self.w_o(x_t))
            hidden = output_gate * h_tilde
            return hidden, (cell_new, normalizer_new, m_new)

    class XLSTMCore(base):  # type: ignore[misc]
        """mLSTM-Stapel mit demselben Aufrufvertrag wie ``nn.GRU``/``nn.LSTM``.

        ``forward(input, hx=None) -> (output, hidden)``, ``batch_first=True`` erzwungen
        (die einzige Aufrufform, die ``torch_sequence.py`` je verwendet). ``hidden`` ist
        ein Tupel von ``num_layers`` Drei-Tupeln ``(C, n, m)`` -- wie bei GRU/LSTM wird es
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
                raise ValueError("XLSTMCore unterstützt nur batch_first=True")
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.layers = nn_module.ModuleList(
                _XLSTMLayer(input_size if index == 0 else hidden_size, hidden_size)
                for index in range(num_layers)
            )
            self.dropout = nn_module.Dropout(dropout)

        def forward(self, sequence: Any, hx: Any | None = None) -> tuple[Any, Any]:
            """Historie oder Einzelschritt kodieren -- Sequenzlänge ist beliebig."""
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
                    hidden, state = layer.step(layer_input[:, step, :], state)
                    outputs.append(hidden)
                states[layer_index] = state
                layer_input = torch_module.stack(outputs, dim=1)
                if layer_index < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            return layer_input, tuple(states)

    # Exposed so that mixed stacks (models/slstm.py) reuse the exact same mLSTM layer.
    XLSTMCore.layer_class = _XLSTMLayer  # type: ignore[attr-defined]
    return XLSTMCore
