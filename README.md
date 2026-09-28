# Generator de text caracter-cu-caracter cu LSTM

Laboratorul implementează un language model autoregresiv PyTorch pentru texte românești. Modelul primește un context de caractere și estimează următorul caracter sau tokenul special `<EOS>`.

## Cerințe și instalare

Este necesar Python 3.10+ și `pip`. Se recomandă un mediu virtual separat.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Pentru GPU, instalează versiunea PyTorch potrivită plăcii video, conform [instrucțiunilor oficiale](https://pytorch.org/get-started/locally/). Codul detectează automat CUDA; în caz contrar rulează pe CPU.

## Rulare

Configurația implicită este în `config/config.yaml`. Parametrii principali care pot fi modificați sunt `sequence_length`, dimensiunile LSTM, `batch_size`, `learning_rate`, numărul de epoci și `patience`.

```powershell
python -m src.train --config config/config.yaml
python -m src.evaluate --config config/config.yaml
python -m src.generate --config config/config.yaml --seed-text "Într-o zi" --temperature 0.8
python -m src.experiment --config config/config.yaml
python -m src.interactive --config config/config.yaml
```

`src.experiment` pornește antrenarea, evaluează o singură dată checkpoint-ul selectat pe test și salvează câte un exemplu pentru temperaturile `0.5`, `0.8`, `1.0` și `1.2`.

### Generare interactivă

`src.interactive` încarcă o singură dată `checkpoints/best_model.pth`, apoi acceptă oricâte texte inițiale din terminal. După fiecare Enter, modelul afișează continuarea; sesiunea se încheie curat prin `Ctrl+C` sau, în PowerShell, `Ctrl+Z` urmat de Enter.

```powershell
# Sampling repetabil cu seed-ul 42 din configurație
python -m src.interactive --config config/config.yaml --temperature 0.8 --max-length 300 --seed-mode fixed

# Sampling variat: se alege și se afișează un seed nou pentru fiecare încercare
python -m src.interactive --config config/config.yaml --temperature 1.0 --max-length 300 --seed-mode random

# Sampling repetabil cu un seed ales explicit
python -m src.interactive --config config/config.yaml --temperature 0.5 --max-length 200 --seed-mode fixed --seed 2026
```

Cu `data.sequence_unit: sentence`, introdu un prefix fără `.`; modelul a fost antrenat să trateze punctul ca final de propoziție, urmat imediat de `<EOS>`. De exemplu, `Într-o zi` este valid, iar `Într-o zi.` este respins cu un mesaj explicativ. În modul `fixed`, același prompt și aceiași parametri produc aceeași predicție; în modul `random`, seed-ul afișat după fiecare generare poate fi reutilizat ulterior cu `--seed-mode fixed --seed VALOARE`.

În terminal, fiecare etapă este anunțată explicit. Pentru fiecare epocă, `tqdm` afișează progresul pe batch-uri și loss-ul batch-ului curent. La finalul epocii se afișează un rezumat în stil Keras/TensorFlow cu durata, `loss`, `perplexity`, `val_loss`, `val_perplexity` și starea checkpoint-ului. Poți dezactiva bara de progres, păstrând rezumatele pe epocă, prin `training.show_progress: false` în configurație.

## Reproductibilitate

`seed` din `config/config.yaml` fixează generatorii Python, NumPy, PyTorch/CPU, PyTorch/CUDA și ordinea de `shuffle` a fiecărui `DataLoader`. Sunt activate și algoritmii PyTorch deterministici. Cu același mediu software, hardware, configurație, corpus și seed, rulările sunt reproductibile. Pe GPU, PyTorch poate raporta explicit o operație fără implementare deterministă, în loc să producă silențios rezultate diferite.

## Date și split

Corpusul `readerbench/ro-stories` este descărcat la prima rulare ca fișierul oficial `ro_paragraphs_12516.csv` din Hugging Face Hub. Fiecare paragraf este păstrat împreună cu cheia operei `(author, title)`. Splitul determinist 80/10/10 se face pe opere înainte de construirea ferestrelor, astfel încât paragrafele aceleiași opere nu apar în subseturi diferite.

Vocabularul este construit exclusiv din train. `<EOS>` este un singur token și nu reprezintă caracterele literale din șirul `"<EOS>"`. Unitatea curentă este definită prin `data.sequence_unit: sentence`: fiecare fragment încheiat cu `.` devine o secvență independentă, punctul este păstrat, iar `<EOS>` este adăugat imediat după acesta. De exemplu, `"Ana pleacă. Dan rămâne."` devine `"Ana pleacă.<EOS>"` și `"Dan rămâne.<EOS>"`. Un fragment final care nu se încheie cu punct este păstrat ca secvență, pentru a nu elimina text din corpus. Regula este intenționat literală: orice `.` este delimitator, inclusiv în abrevieri sau numere.

Pentru a reveni la vechiul comportament, setează `data.sequence_unit: paragraph`. După orice schimbare a acestei opțiuni trebuie relansată antrenarea: checkpoint-urile și rezultatele anterioare sunt incompatibile semantic cu noua segmentare. `<UNK>` mapează caractere nevăzute în validation, test sau seed. Ferestrele nu traversează granițele secvențelor.

## Antrenare și artefacte

Modelul are arhitectura `Embedding → LSTM → Linear` și produce logits brute. Antrenarea utilizează `CrossEntropyLoss`, Adam, gradient clipping și early stopping pe validation loss. După antrenare este încărcat automat checkpoint-ul cu cel mai mic validation loss.

Sunt salvate următoarele rezultate:

- `checkpoints/best_model.pth`: checkpoint-ul ales pe validation.
- `outputs/vocabulary.json`: mapping-ul persistent al tokenurilor.
- `outputs/split_manifest.json`: operele atribuite fiecărui subset.
- `outputs/training_history.json`: loss și perplexity pe epocă.
- `outputs/training_summary.json`: cea mai bună epocă, durata antrenării, statistici ale splitului și numărul de parametri.
- `outputs/test_metrics.json`: evaluarea finală pe test.
- `outputs/temperature_samples.json`: exemplele comparative de generare.
- `outputs/config_used.yaml`: configurația folosită în experiment.
- `outputs/plots/training_loss.png`: curbele train/validation pentru cross-entropy.
- `outputs/plots/training_perplexity.png`: curbele train/validation pentru perplexity.
- `outputs/plots/split_statistics.png`: numărul de opere, paragrafe și caractere din fiecare subset.
- `outputs/plots/final_test_metrics.png`: loss și perplexity ale checkpoint-ului pe test.
- `outputs/plots/temperature_lengths.png`: lungimile generate până la EOS pentru fiecare temperatură.

Perplexitatea este calculată ca $PPL = e^L$, unde $L$ este cross-entropy-ul mediu pe tokenurile țintă.

`training_summary.json` include și rata caracterelor necunoscute în validation/test. Aceasta permite interpretarea corectă a evaluării atunci când vocabularul este construit exclusiv din train. Pentru mostrele generate se salvează suplimentar numărul de caractere distincte, raportul de diversitate și numărul de perechi adiacente repetitive.

## EOS și temperatură

`sequence_length` controlează doar lungimea ferestrelor de antrenare. La generare, modelul eșantionează autoregresiv din $softmax(logits / T)$ până când eșantionează `<EOS>`. `max_generation_length` este doar o limită de siguranță pentru evitarea unei bucle infinite; nu stabilește lungimea normală a textului generat.

Pentru analiza obligatorie, păstrează același seed text și compară mostrele pentru cele patru valori de temperatură. Notează pentru fiecare lungimea produsă până la EOS, coerența locală, diversitatea, opririle premature și repetițiile. În varianta cu EOS, lungimea generată este o variabilă aleatoare influențată de distribuția modelului și de sampling, spre deosebire de varianta care ar genera exact $N$ caractere.

## Teste

```powershell
python -m pytest -q
```

Testele verifică vocabularul train-only, EOS, splitul pe opere, ferestrele independente, formele modelului, checkpoint-ul și oprirea la EOS.