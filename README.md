# Forecasting Pipeline

Automatizovaný rámec na trénovanie, predikciu a vyhodnotenie časových radov pomocou viacerých modelov hlbokého učenia.

---

## Obsah

- [Štruktúra projektu](#štruktúra-projektu)
- [Inštalácia](#inštalácia)
- [Modely](#modely)
- [Spustenie jednotlivého modelu](#spustenie-jednotlivého-modelu)
- [Spustenie pipeline](#spustenie-pipeline)
- [Spustenie experimentov](#spustenie-experimentov)
- [Výstupy](#výstupy)
- [Metriky](#metriky)

---

## Štruktúra projektu

```
.
├── Data/
│   ├── weatherHistory.csv
│   └── ETTh1.csv
├── Models/
│   ├── TFT/
│   │   └── tft.py
│   ├── DeepAR/
│   │   └── deepAR.py
│   ├── NBeats/
│   │   └── NBeats.py
│   ├── Tsmixer/
│   │   └── tsmixer.py
│   └── LTSF-Linear/
│       └── run_longExp.py
├── Pipeline/
│   ├── pipeline.py
│   ├── run_experiments.py
│   ├── evaluate.py
│   └── outputs/
│       ├── experiments.csv
│       └── <experiment_id>/
│           ├── results.csv
│           └── <model>_output.txt
└── requirements.txt
```

---

## Inštalácia

> **TODO:** `requirements.txt` nie je ešte finalizovaný.

```bash
pip install -r requirements.txt
```

---

## Modely

| Model | Skript |
|---|---|
| TFT (Temporal Fusion Transformer) | `Models/TFT/tft.py` |
| DeepAR | `Models/DeepAR/deepAR.py` |
| N-BEATS | `Models/NBeats/NBeats.py` |
| TSMixer | `Models/Tsmixer/tsmixer.py` |
| LTSF-Linear (DLinear) | `Models/LTSF-Linear/run_longExp.py` |


Každý model prijíma rovnaké argumenty:

| Argument | Popis |
|---|---|
| `--train-dataset` | cesta k trénovacej množine (škálovaná) |
| `--val-dataset` | cesta k validačnej množine (škálovaná) |
| `--test-dataset` | cesta k testovacej množine (škálovaná) |
| `--target` | názov cieľového stĺpca |
| `--date` | názov stĺpca s dátumom |
| `--horizon` | počet krokov predikcie dopredu |
| `--lookback-window` | dĺžka vstupného okna |
| `--seed` | seed pre reprodukovateľnosť |
| `--output` | cesta k výstupnému súboru (JSON) |

Výstupmi každého modelu sú txt súbor s metadátami (aktuálne obsahuje epochu, na ktorej skončilo trénovanie) a csv súbor s predikovanými hodnotami

---

## Spustenie jednotlivého modelu

Model možno spustiť priamo, no je potrebné manuálne pripraviť rozdelené a škálované datasety. Odporúča sa spúšťať cez `pipeline.py`.

Príklad priameho spustenia TFT:

```bash
python3 Models/TFT/tft.py \
    --train-dataset Data/train.csv \
    --val-dataset   Data/val.csv \
    --test-dataset  Data/test.csv \
    --target        "Temperature (C)" \
    --date          "Formatted Date" \
    --horizon       24 \
    --lookback-window 96 \
    --seed          42 \
    --output        output.txt
```

---

## Spustenie pipeline

`pipeline.py` zabezpečuje:
1. načítanie a chronologické rozdelenie datasetu (70 % train / 15 % val / 15 % test)
2. normalizáciu pomocou `StandardScaler` fitovaného **iba na trénovacích dátach**
3. spustenie všetkých aktívnych modelov na škálovaných dátach
4. odškálovanie predikcií späť na pôvodné hodnoty (inverse transform)
5. vyhodnotenie predikcií oproti pôvodnému (neškálovanému) testovaciemu setu

```bash
python3 Pipeline/pipeline.py \
    --dataset         Data/weatherHistory.csv \
    --target          "Temperature (C)" \
    --date            "Formatted Date" \
    --horizon         24 \
    --lookback-window 96 \
    --seed            42 \
    --output-dir      Pipeline/outputs/1
```

### Parametre pipeline

| Argument | Povinný | Default | Popis |
|---|---|---|---|
| `--dataset` | áno | — | cesta k CSV datasetu |
| `--target` | áno | — | názov cieľového stĺpca |
| `--date` | nie | `"date"` | názov stĺpca s dátumom |
| `--horizon` | áno | — | počet krokov predikcie |
| `--lookback-window` | nie | `4 × horizon` | dĺžka vstupného okna |
| `--seed` | nie | náhodný | seed pre reprodukovateľnosť |
| `--output-dir` | nie | `Pipeline/outputs` | priečinok pre výstupy |

---

## Spustenie experimentov

`run_experiments.py` spúšťa `pipeline.py` pre každú kombináciu datasetu a horizontu definovanú v zozname `EXPERIMENTS` v tom istom súbore.

```bash
python3 Pipeline/run_experiments.py
```

Konfigurácia experimentov sa upravuje priamo v súbore `run_experiments.py`:

```python
EXPERIMENTS = [
    {
        "dataset":  "Data/weatherHistory.csv",
        "target":   "Temperature (C)",
        "date":     "Formatted Date",
        "horizons": [24, 48],
        "lookback-window": 96,
        "seed": 42,
    },
    {
        "dataset":  "Data/ETTh1.csv",
        "target":   "OT",
        "date":     "date",
        "horizons": [24, 48, 96],
    },
]
```

Každý experiment dostane unikátne ID a jeho stav (`running` / `success` / `failed`) sa priebežne zapisuje do `Pipeline/outputs/experiments.csv`.

---

## Výstupy

### `Pipeline/outputs/experiments.csv`

Register všetkých spustených experimentov:

| Stĺpec | Popis |
|---|---|
| `id` | unikátne ID experimentu |
| `dataset` | cesta k datasetu |
| `target` | cieľový stĺpec |
| `date` | stĺpec s dátumom |
| `horizon` | predikčný horizont |
| `results_file` | cesta k `results.csv` daného experimentu |
| `status` | `running` / `success` / `failed` |
| `started_at` | čas spustenia |

### `Pipeline/outputs/<id>/results.csv`

Výsledky metrík pre každý model v danom experimente:

| Stĺpec | Popis |
|---|---|
| `Dataset` | názov zdrojového datasetu |
| `Target` | cieľový stĺpec |
| `Model` | názov modelu |
| `Horizon` | predikčný horizont |
| `Lookback window` | dĺžka vstupného okna |
| `Seed` | použitý seed |
| `Train Time (s)` | čas trénovania v sekundách |
| `MSE` | stredná kvadratická chyba |
| `MAE` | stredná absolútna chyba |
| `MAPE (%)` | stredná absolútna percentuálna chyba |
| `DA (%)` | smerová presnosť |

---

## Metriky

Všetky metriky sa počítajú oproti **pôvodným neškálovaným hodnotám** testovacej množiny.

**MSE** — penalizuje väčšie odchýlky kvadraticky, citlivejšia na odľahlé hodnoty:
$$\text{MSE} = \frac{1}{h} \sum_{i=1}^{h} (y_i - \hat{y}_i)^2$$

**MAE** — priemerná veľkosť chyby v pôvodných jednotkách:
$$\text{MAE} = \frac{1}{h} \sum_{i=1}^{h} |y_i - \hat{y}_i|$$

**MAPE** — chyba relatívne voči skutočným hodnotám (chránená proti deleniu nulou hodnotou $\varepsilon = 10^{-8}$):
$$\text{MAPE} = \frac{100}{h} \sum_{i=1}^{h} \left| \frac{y_i - \hat{y}_i}{\max(|y_i|, \varepsilon)} \right|$$

**DA (Directional Accuracy)** — percento krokov kde predikcia správne odhadla smer zmeny (rast/pokles) voči predchádzajúcej skutočnej hodnote. Pri horizonte 1 nie je definovaná (`N/A`).
