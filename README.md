# Analiza emisije CO₂ 🚗💨

Emisija ugljen-dioksida (CO₂) iz saobraćaja jedan je od glavnih faktora koji doprinose globalnom zagrevanju i zagađenju vazduha. Na količinu emitovanog CO₂ utiču različiti faktori, uključujući tip vozila, potrošnju goriva i vrstu goriva. Zbog toga je analiza emisija ključna za razumevanje i smanjenje negativnog uticaja saobraćaja na životnu sredinu.

Ovaj projekat se fokusira na analizu podataka o potrošnji goriva i emisiji CO₂ kako bi se identifikovali ključni faktori koji doprinose povećanoj emisiji. Korišćenjem tehnika obrade podataka, mašinskog učenja i vizualizacije, cilj je da se pruže korisni uvidi, uključujući predikciju emisije CO₂ i klasterizaciju vozila na osnovu njihovih karakteristika.

## 🛠 Korišćene tehnologije

    🐍 Python 3.12.9
    📊 Pandas, NumPy, Matplotlib – Obrada i vizualizacija podataka
    🤖 Scikit-learn – Analiza podataka i mašinsko učenje
    🌿 Flask – Web framework

## 📂 Skup podataka

Za analizu emisije CO₂ korišćen je skup podataka ***Vehicle CO2 Emissions Dataset*** preuzet sa https://www.kaggle.com/datasets/brsahan/vehicle-co2-emissions-dataset/data.

## ⭐ Ključne karakteristike  

- **Brend:** Proizvođač vozila (npr. Toyota, Ford, BMW).

- **Tip vozila:** Klasifikacija vozila na osnovu veličine i namene (npr. SUV, Sedan).

- **Zapremina motora (L):** Radna zapremina motora izražena u litrima.

- **Broj cilindara:** Broj cilindara u motoru.
  
- **Menjač:** Tip menjača (npr. Automatski, Manuelni).

- **Tip goriva:** Vrsta goriva koje vozilo koristi (npr. Benzin, Dizel, Hibrid).

- **Potrošnja goriva (grad, autoput i kombinovano):** Efikasnost potrošnje goriva izražena u litrima na 100 kilometara (L/100 km).

- **Emisija CO₂ (g/km):** Emisija ugljen-dioksida po pređenom kilometru (ciljna promenljiva za predikciju).

## 💻 Pokretanje projekta

### Kloniranje repozitorijuma:
   
```sh
git clone https://github.com/nemanjaASE/fuel-consumption.git
cd fuel-consumption
```

### Kreiranje foldera za podatke:
   
```sh
mkdir data
# co2.csv smestiti u folder /data
```

### Kreiranje i aktivacija virtualnog okruženja
   
Windows (cmd/Powershell):
```sh
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:
```sh
python3 -m venv venv
source venv/bin/activate
```

### Instalacija potrebnih paketa i pokretanje

```sh
pip install -r requirements.txt
python app.py
```

🌍 Aplikacija pokrenuta na http://127.0.0.1:5000/

![Screenshot 2025-03-03 100917](https://github.com/user-attachments/assets/e2c50f57-618e-46a5-9a9f-94cf8b7a19c0)

## Predikcija emisije CO₂

Izborom opcije ***Calculate Emissions*** prelazi se na formu za unos svojstva na osnovu kojih se vrši predikcija emisije CO₂.

![Screenshot 2025-03-03 102608](https://github.com/user-attachments/assets/88d7a7db-2ebd-404c-abf5-aced7c3189b8)

Za uneta svojstva dobija se sledeći rezultat:

![Screenshot 2025-03-03 121626](https://github.com/user-attachments/assets/149fadd5-ecee-4724-b1fc-7bda3aa2b8de)

## Provera ekonomičnosti vozila

Izborom opcije ***Check Vehicle Economy*** prelazi se na formu za unos svojstva na osnovu kojih se vrši provera ekonomičnosti vozila.

![Screenshot 2025-03-03 122749](https://github.com/user-attachments/assets/6538a0e7-583f-4fa2-a9d4-677581de2833)

Za uneta svojstva dobija se sledeći rezultat:

![Screenshot 2025-03-03 122814](https://github.com/user-attachments/assets/bbdc49d5-ce85-40aa-9d6d-2dbaee70d293)

## Analiza podataka

Izborom opcije ***Explore Data Trends*** prelazi se na vizualni prikaz emisije CO₂ za svaku marku automobila.

![Screenshot 2025-03-03 123507](https://github.com/user-attachments/assets/d86205f6-3d5d-4563-9d9b-91733f4b22af)

Omogućen je i izbor marke automobile na osnovu čega se vrši proračun koja svojstva i u kom udelu  utiču na emisiju CO₂. Dat je primer za ***Volkswagen*** automobile.

![Screenshot 2025-03-03 123812](https://github.com/user-attachments/assets/0d1d8908-442c-4143-b2a2-94d132d50ec2)

# Treniranje modela za analizu emisije CO₂

### 1️⃣ Priprema podataka 

📌 Učitavanje podataka iz data/co2.csv

📌 Čišćenje podataka i rukovanje nedostajućim vrednostima

📌 Skaliranje numeričkih podataka

### 2️⃣ Odabir modela

📌 Koriste se algoritmi poput linearne regresije, random forest-a ili XGBoost-a

### 3️⃣ Treniranje i evaluacija

📌 Podela podataka na trening (80%) i test (20%) skup

📌 Treniranje modela na trening podacima

📌 Evaluacija modela pomoću R², MAE, RMSE metrika

### 4️⃣ Testiranje i predikcija

📌 Model se testira na novim podacima

📌 Predikcija emisije CO₂ na osnovu input karakteristika vozila

## Odabir modela

Kako bi se pronašao najprecizniji model za predikciju emisije CO₂ na osnovu podataka o potrošnji goriva i karakteristikama vozila, testirano je više modela mašinskog učenja. Proces odabira modela sastojao se iz sledećih koraka:

### 1️⃣ Definisanje ulaznih podataka

📌 Kao prediktori (feature-i) izabrane su numeričke karakteristike vozila koje su pokazale visoku korelaciju sa emisijom CO₂:

    Potrošnja goriva u gradu, na autoputu i kombinovana potrošnja
    Zapremina motora
    Broj cilindara
    Tip goriva (enkodiran korišćenjem One-Hot Encoding) 
    
📌 Ciljna promenljiva (target) bila je CO₂ Emissions (g/km).

Korelaciju svih promenljivih sa emisijom CO₂ je izračunata koristeći Pearsonovu korelaciju.

![Screenshot 2025-03-02 115958](https://github.com/user-attachments/assets/3942a3f6-4642-433c-aa90-99e96cac8773)

### 2️⃣ Podela podataka

📌 Podaci su podeljeni u trening (80%) i test skup (20%), koristeći train_test_split iz biblioteke sklearn.

### 3️⃣ Korišćeni modeli

Testirani su sledeći modeli regresije:

    Linear Regression – osnovni model koji procenjuje linearne odnose između varijabli.
    Lasso Regression – verzija linearne regresije sa L1 regularizacijom koja eliminiše manje značajne karakteristike.
    Ridge Regression – linearni model sa L2 regularizacijom koji smanjuje problem prekomernog uklapanja (overfitting).
    Decision Tree – drvo odlučivanja koje segmentira podatke na osnovu pravila.
    Random Forest – ansambl metoda koja koristi više stabala odlučivanja radi bolje generalizacije.
    Gradient Boosting – model koji iterativno poboljšava predikcije kombinujući jednostavna stabla odlučivanja.
    XGBoost – optimizovana verzija gradient boosting-a, poznata po visokoj preciznosti.

### 4️⃣ Evaluacija performansi modela

Za ocenjivanje tačnosti modela korišćene su sledeće metrike:

    MAE (Mean Absolute Error) – prosečna apsolutna greška predikcije.
    MSE (Mean Squared Error) – prosečna kvadratna greška (kažnjava veće greške više od MAE).
    RMSE (Root Mean Squared Error) – koren MSE, interpretabilniji jer je u istim jedinicama kao ciljna promenljiva.
    R² Score – pokazuje koliko model objašnjava varijabilnost ciljne promenljive (što bliže 1, to bolje).
    Cross-validation R² – prosek R² skora dobijenog kros-validacijom sa 5 preklopa (KFold).

### 5️⃣ Odabir najboljeg modela

📌 Modeli su testirani koristeći 5-fold kros-validaciju kako bismo dobili stabilnije rezultate.

📌 Pored preciznosti, analizirano je i vreme treniranja i predikcije, što je bitno za upotrebu modela u realnim scenarijima.

## 📊 Performanse modela

| Model               | MAE   | MSE    | RMSE  | R² Score | Mean CV R² | Train Time (s) | Predict Time (s) |
|---------------------|-------|--------|-------|----------|------------|----------------|------------------|
| Linear Regression  | 3.2053 | 39.7522 | 6.3049 | 0.9884   | 0.9915     | 0.0020         | 0.0000           |
| Lasso Regression   | 3.6543 | 42.8701 | 6.5475 | 0.9875   | 0.9901     | 0.0048         | 0.0000           |
| Ridge Regression   | 3.2417 | 39.4022 | 6.2771 | 0.9885   | 0.9915     | 0.0000         | 0.0000           |
| Decision Tree      | 2.2285 | 17.4361 | 4.1757 | 0.9949   | 0.9961     | 0.0160         | 0.0000           |
| Random Forest     | 2.2045 | 15.0339 | 3.8774 | 0.9956   | 0.9967     | 0.2554         | 0.0078           |
| Gradient Boosting  | 2.4760 | 16.8522 | 4.1051 | 0.9951   | 0.9968     | 0.1766         | 0.0036           |
| XGBoost           | 2.5280 | 18.1204 | 4.2568 | 0.9947   | 0.9964     | 0.0318         | 0.0040           |

### 📢 Zaključak:
- **Najprecizniji model**: *Random Forest* (najmanji MAE, MSE, RMSE, visok R²).  
- **Najbrži model**: *Linear Regression* (najkraće vreme treniranja i predikcije).  
- **Najstabilniji model**: *Gradient Boosting* (visok Mean CV R²).

Na osnovu rezultata, ***Random Forest*** model sa najboljim balansom između tačnosti i brzine izvršavanja odabran je za finalnu implementaciju.

Nakon treniranja, model i svojstva se čuvaju u models/ direktorijumu:

    co2_emission_model.pkl
    feature_columns.pkl

# Treniranje modela za klasterizaciju

U ovom projektu koristi se ***KMeans algoritam*** za grupisanje vozila na osnovu njihovih karakteristika potrošnje goriva i emisije CO₂.

### 📌 Podaci

Za treniranje modela koristi se dataset ***co2***.csv, a sledeće osobine su odabrane za klasterovanje:

    Fuel Consumption City (L/100 km)
    Fuel Consumption Comb (L/100 km)
    Fuel Consumption Hwy (L/100 km)
    Engine Size (L)
    Cylinders
    CO2 Emissions (g/km)

### 🛠️ Treniranje Modela

Prilikom treniranja podaci se prvo standardizuju (StandardScaler), zatim se koristi KMeans algoritam sa:

    3 klastera (n_clusters=3)
    Inicijalizacija pomoću k-means++
    Maksimalno 300 iteracija (max_iter=300)
    10 ponovnih inicijalizacija (n_init=10)
    Random state = 42 za reproduktivne rezultate

Nakon treniranja, model i skalirani podaci se čuvaju u models/ direktorijumu:

    kmeans_model.pkl – Sačuvani model klasterovanja
    scaler.pkl – Standardizovani objekat za skaliranje podataka

# Zaključak

Ovaj projekat pruža sveobuhvatnu analizu potrošnje goriva i emisije CO₂ korišćenjem različitih metoda mašinskog učenja, uključujući regresione modele za predikciju emisije CO₂ i KMeans klasterovanje za grupisanje vozila na osnovu njihovih karakteristika.

Kroz implementaciju višestrukih modela (Linear Regression, Random Forest, Gradient Boosting itd.), postignuta je visoka tačnost u predviđanju emisije CO₂, dok je klasterovanjem omogućena segmentacija vozila prema efikasnosti potrošnje goriva. Projekat takođe uključuje standardizaciju podataka, evaluaciju modela i čuvanje treniranih modela radi kasnije primene.

Ovaj projekat može poslužiti kao osnova za dalja istraživanja i razvoj alata koji bi pomogli industriji, regulatorima i krajnjim korisnicima da donose bolje odluke o potrošnji goriva i smanjenju emisije štetnih gasova.
