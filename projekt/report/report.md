# Report – Detekcia stresu z rečového signálu (TESS + WS3D)

## 1. Formulácia problému
Cieľom projektu je navrhnúť a implementovať systém na **detekciu stresu z rečového signálu** pomocou neurónových sietí. Stres je klasifikovaný binárne ako:
- **0 – no-stress**
- **1 – stress**

Projekt porovnáva dva scenáre:
1. **Nepriama detekcia stresu (TESS)** – stres je aproximovaný z emočného prejavu v reči (určité emócie mapované na „stress“).
2. **Priama detekcia stresu (WS3D / WorkStress3D)** – stres/no-stress je odvodený z označenia/prefixu v názve nahrávky (mapovanie definované v konfigurácii projektu).

Pre oba datasety porovnávame dve architektúry:
- **MLP** (Multi-Layer Perceptron) nad vektorom príznakov (features),
- **CNN** (Convolutional Neural Network) nad log-Mel spektrogramami.

---

## 2. Teoretický background
### 2.1 Stres v reči a reprezentácia signálu
Stres sa v reči prejavuje napr. zmenami v:
- prosódii (tempo, intonácia),
- energii a hlasitosti,
- spektrálnych vlastnostiach (zmena distribúcie energie vo frekvenciách).

Na spracovanie reči sa často používajú:
- **MFCC** (Mel-Frequency Cepstral Coefficients) – kompaktná reprezentácia spektra relevantná pre percepciu,
- **log-Mel spektrogram** – časovo-frekvenčná reprezentácia vhodná pre CNN.

### 2.2 MLP nad príznakmi
MLP je vhodné, keď máme už extrahované príznaky (features). Výhodou je jednoduchosť a nízke výpočtové nároky, nevýhodou citlivosť na kvalitu features a stratu informácie, ak príliš agregujeme čas.

### 2.3 CNN nad spektrogramom
CNN vie učiť lokálne vzory v časovo-frekvenčnom obraze (spektrálne formanty, zmeny energie). Výhodou je schopnosť učiť sa robustné reprezentácie priamo z „obrazu“ spektrogramu.

### 2.4 Klasifikačné metriky
Pre binárnu klasifikáciu používame:
- **Accuracy (ACC)** – podiel správnych klasifikácií,
- **Precision, Recall, F1-score** – vhodné pri nevyvážených triedach,
- **ROC-AUC** – robustná metrika pri prahovaní pravdepodobností,
- **Confusion matrix** – pre analýzu typov chýb.

---

## 3. Popis datasetu a predspracovanie dát
### 3.1 TESS
TESS (Toronto Emotional Speech Set) obsahuje nahrávky hovoreného slova s rôznymi emóciami. Pre účely projektu sa emócie mapujú na binárny stres:
- `neutral`, `calm`, `ps` → **0 (no-stress)**
- `angry`, `fear`, `disgust`, `sad` → **1 (stress)**

**Split:** v implementácii sa používa stratifikovaný split na train/val/test (pomery podľa konfigurácie).

### 3.2 WS3D (WorkStress3D)
WS3D obsahuje reálne nahrávky, kde sa labely odvodzujú z prefixu v názve nahrávky podľa mapovania definovaného v `Config.WS3D_LABEL_MAPPING`:
- `a` (angry) → 1
- `n` (neutral) → 0
- `h` (happy) → 0
- `sa` (sadness) → 1
- `d` (disgust) → 1
- `f` (fear) → 1
- `ps` (pleasant surprise) → 0
- `c` (calm) → 0

**Split:** speaker-independent (subjekty sa medzi train/val/test nemiešajú), aby sa predišlo úniku informácie.

### 3.3 Predspracovanie
Predspracovanie zahŕňa:
- resampling na **16 kHz**,
- zarovnanie/orez/padding na fixnú dĺžku (napr. 3 s),
- extrakciu features:
  - pre MLP: MFCC (+ prípadne delta/delta2 podľa vášho spracovania),
  - pre CNN: log-Mel spektrogram.

Výstupy sa ukladajú do:
- `data/processed/<dataset>/features/*.npy`
- `data/processed/<dataset>/spectrograms/*.npy`

---

## 4. Metodológia
### 4.1 Vstupy modelov
- **MLP**: vstup je vektor príznakov `x ∈ R^D` (D závisí od datasetu/feature pipeline).
- **CNN**: vstup je log-Mel spektrogram `x ∈ R^{1×N_MELS×T}`.

### 4.2 Tréning
Použité prvky tréningu:
- loss: **BCEWithLogitsLoss** (binárna klasifikácia),
- optimizer: napr. Adam,
- regularizácia: weight decay, dropout,
- early stopping / výber best checkpointu podľa validačnej metriky (ak implementované v Trainer-i).

### 4.3 Augmentácie
Augmentácie sa používajú len pre train:
- pre MLP: **GaussianNoise**,
- pre CNN: **SpectrogramNoise** a **SpecAugment** (časové a frekvenčné maskovanie).

---

## 5. Návrh experimentov
Navrhli sme experimenty tak, aby porovnávali:
1. Datasetový efekt: TESS vs WS3D
2. Modelový efekt: MLP vs CNN
3. Dopad augmentácií (ablation study – kontrolovaný experiment)

### 5.1 Základné experimenty
- TESS + MLP
- TESS + CNN
- WS3D + MLP
- WS3D + CNN

### 5.2 Ablation study (kontrolovaný experiment)
Minimálne jeden kontrolovaný experiment:
- **CNN na WS3D s augmentáciami vs bez augmentácií**
  - (A) WS3D+CNN s `SpecAugment` + `SpectrogramNoise`
  - (B) WS3D+CNN bez augmentácií (transform=None)

Kontrolované je všetko okrem augmentácií (rovnaké epochy, batch size, optimizer, split, seed).

---

## 6. Výsledky
### 6.1 Tabuľka výsledkov (test set)
Doplňte hodnoty z `outputs/summary_metrics.json` a z test evaluácie.

| Dataset | Model | ACC | F1 | Precision | Recall | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| TESS | MLP | [DOPLŇ] | [DOPLŇ] | [DOPLŇ] | [DOPLŇ] | [DOPLŇ] |
| TESS | CNN | [DOPLŇ] | [DOPLŇ] | [DOPLŇ] | [DOPLŇ] | [DOPLŇ] |
| WS3D | MLP | [DOPLŇ] | [DOPLŇ] | [DOPLŇ] | [DOPLŇ] | [DOPLŇ] |
| WS3D | CNN | [DOPLŇ] | [DOPLŇ] | [DOPLŇ] | [DOPLŇ] | [DOPLŇ] |

### 6.2 Priebeh tréningovej a validačnej chyby
Do reportu vložte grafy generované skriptom (napr. `*_training_curves.png`):
- train loss vs val loss,
- train accuracy vs val accuracy.

Komentár (čo sledovať):
- či val loss klesá podobne ako train loss (generalizácia),
- či sa objavuje overfitting (train sa zlepšuje, val sa zhoršuje),
- stabilita tréningu (oscilácie).

### 6.3 Ablation study – výsledky
| Experiment | ACC | F1 | ROC-AUC | Poznámka |
|---|---:|---:|---:|---|
| WS3D + CNN (augmentácie ON) | [DOPLŇ] | [DOPLŇ] | [DOPLŇ] | SpecAugment + noise |
| WS3D + CNN (augmentácie OFF) | [DOPLŇ] | [DOPLŇ] | [DOPLŇ] | transform=None |

---

## 7. Kritická analýza a diskusia
### 7.1 Porovnanie MLP vs CNN
Očakávania:
- CNN môže lepšie zachytiť časovo-frekvenčné vzory súvisiace so stresom (ak je dosť dát),
- MLP môže byť stabilnejšie pri menšom datasete alebo pri kvalitných features.

Diskusia (doplňte podľa výsledkov):
- Na ktorom datasete vyhralo CNN a prečo?
- Kde zlyhalo (overfitting, málo dát, šum, nekonzistentné labely)?

### 7.2 Rozdiel medzi TESS a WS3D
TESS je emočný dataset a stres je len aproximácia – model sa učí skôr „emočný stres“ než fyziologický/psychologický stres.
WS3D je realistickejší, no môže mať:
- väčšiu variabilitu nahrávok,
- šum, rôzne zariadenia,
- nevyvážené triedy,
- riziko „slabého“ labelovania, ak je label odvodený z názvu a nie z priamo anotovaného stresu.

### 7.3 Analýza chýb (confusion matrix)
Do reportu vložte confusion matrix pre každú kombináciu.
Odporúčaná analýza:
- či model viac robí FP (no-stress → stress) alebo FN (stress → no-stress),
- čo je horšie z pohľadu aplikácie (typicky FN je rizikovejší).

---

## 8. Obmedzenia a možnosti ďalšej práce
### Obmedzenia
- WS3D labelovanie z prefixu nemusí priamo zodpovedať „stresu“ v reálnej situácii (môže ísť o emócie).
- Obmedzený počet subjektov → zložitejšia generalizácia v speaker-independent splite.
- Fixná dĺžka segmentu (napr. 3 s) môže zahodiť kontext.

### Ďalšia práca
- Zaviesť robustnejší preprocessing (VAD – voice activity detection).
- Skúsiť architektúry pre audio: CRNN, wav2vec2 embeddings + klasifikátor.
- Lepšie vyváženie tried (weighted loss, focal loss).
- Viac ablation experimentov: bez noise, bez SpecAugment, rôzne N_MELS/N_MFCC.

---

## 9. Prínos jednotlivých členov tímu
(Doplňte mená a konkrétne úlohy.)

- Člen A: preprocessing WS3D, konverzia formátov, príprava features/spektrogramov
- Člen B: implementácia modelov (MLP, CNN), tréningový pipeline
- Člen C: experimenty, vyhodnotenie, grafy, report

---

## 10. Zoznam použitej literatúry
- Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition*. arXiv:1904.08779.
- Základná literatúra ku MFCC a log-Mel reprezentáciám (učebnicové zdroje / dokumentácia knižníc, ktoré používate).
- Dokumentácia PyTorch (Dataset, DataLoader, optimizéry).