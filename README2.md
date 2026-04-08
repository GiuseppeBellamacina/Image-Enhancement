# 📸 Image Enhancement: Approfondimento sulle Degradazioni e Obiettivi del Progetto

Questo documento approfondisce come viene gestita la degradazione delle immagini all'interno del progetto, quale risultato visivo si ottiene, e descrive in dettaglio l'obiettivo primario e le tecniche utilizzate.

---

## 🔍 Obiettivo del Progetto

L'obiettivo principale di questo progetto è **studiare e confrontare diversi metodi di Image Enhancement** avvalendosi di architetture di Deep Learning basate su Reti Neurali Convoluzionali (CNN). 
L'approccio prevede una pipeline end-to-end strutturata nei seguenti passaggi:
1. **Corruzione controllata**: Degradare immagini ad alta risoluzione (dataset DIV2K) in maniera artificiale, governando precisi parametri matematici.
2. **Restauro (Restoration)**: Addestrare modelli avanzati per ripristinare il più possibile i dettagli, la qualità e la brillantezza originali dell'immagine pulita.
3. **Valutazione (Evaluation)**: Confrontare i risultati ottenuti mediante metriche quantitative (come PSNR e SSIM) e misurazioni qualitative percepite dall'occhio umano.

Il fine ultimo è quello di osservare come le diverse architetture (CNN, Attention, Residual) e le funzioni di loss (L1, Perceptual) si comportino di fronte a tipologie e intensità di rumore differenti, valutandone efficacia, stabilità e tempi di convergenza.

---

## 🛠️ Tecniche Utilizzate

Per affrontare il restauro delle immagini, il progetto implementa una suite di tecniche avanzate:

### Architetture di Deep Learning
- **UNet Standard**: Un'architettura encoder-decoder classica con "skip connections". Permette alla rete di mantenere le informazioni ad alta frequenza (i dettagli spaziali) passandole direttamente dall'encoder al decoder. Il modello predice l'immagine pulita in modo diretto.
- **UNet Residual**: Sfrutta il principio del "residual learning". Anziché far predire alla rete l'immagine pulita, le si fa predire la "mappa del rumore" (noise residual). L'immagine finale si ottiene sottraendo il rumore predetto dall'immagine degradata. Converge più rapidamente ed è notoriamente più efficace per task specifici come il denoising.
- **Attention UNet**: Espande l'architettura base integrando dei meccanismi di attenzione (*Attention Gates*) sulle skip connections. Questo permette alla rete di concentrarsi selettivamente sulle porzioni "rilevanti" o maggiormente danneggiate dell'immagine, sopprimendo il rumore di fondo.

### Funzioni di Loss (Loss Functions)
- **CombinedLoss (L1 + SSIM)**: Una loss ibrida. La **Loss L1** (Mean Absolute Error) assicura una convergenza pixel per pixel robusta contro gli outlier. A questa viene affiancata la **Loss SSIM** (Structural Similarity) che spinge la rete a preservare le strutture visive, i contorni e la luminanza, massimizzando la somiglianza strutturale.
- **CombinedPerceptualLoss (L1 + SSIM + VGG)**: Aggiunge la *Perceptual Loss*, che processa le immagini tramite una rete VGG16 pre-addestrata per calcolare l'errore a livello di "feature maps" profonde. Questo vincola il modello a generare dettagli più realistici e texture complesse e visivamente più "corrette" all'occhio umano.

### Tecniche di Training e Validazione
- **Mixed Precision (AMP)** e **Gradient Clipping**: Usati per velocizzare il training (utilizzando calcoli a 16 e 32 bit) e per mantenerlo numericamente stabile, prevenendo l'esplosione dei gradienti.
- **Scheduler e Regolarizzazione**: Utilizzo di un Warmup lineare iniziale seguito da uno scheduler `CosineAnnealing` per il learning rate, combinato con un meccanismo di Early Stopping per fermare il training se il modello inizia ad andare in overfitting.
- **Sliding Window Inference**: In fase di inferenza (su immagini 2K full-resolution), il modello prevede patch più piccole (es. 128x128) che scorrono sull'intera immagine sovrapponendosi (`overlap`). Viene poi applicato un *blending* pesato per riunire i patch senza lasciare artefatti a blocchi o cuciture visibili.

---

## 🧪 Gestione della Degradazione

La degradazione è il punto di partenza per addestrare i modelli a riconoscere e rimuovere il rumore.

### Come viene gestita a livello di Pipeline
La degradazione **non avviene ad ogni epoca in tempo reale (on-the-fly)**, ma viene **pre-computata**. 
Uno script dedicato (`generate_degraded_dataset.py`) prende il dataset originale (DIV2K) e ne genera una copia interamente corrotta basata sui parametri scelti dall'utente. Queste immagini vengono salvate permanentemente su disco in percorsi stratificati e categorizzati (es. `data/degraded/gaussian/sigma_100/`). 
In questo modo, durante le epoche di addestramento, la rete vedrà sempre **la stessa identica** immagine degradata per quello specifico campione. Questo permette di isolare e concentrarsi sulla convergenza del modello. Se si desidera variare il rumore per generalizzare maggiormente, occorre preventivamente generare nuovi parametri e varianti degradate.

### Casuale o a Pattern Fisso?
Dipende dalla tipologia di algoritmo applicata. 

Va precisato che lo script assegna un **seed di generazione univoco per ogni singola immagine** (calcolato come `seed_base + indice_immagine`). Questo trucco assicura che il rumore o il pattern casuale sia **diverso da un'immagine all'altra**, evitando che il modello impari per errore a rimuovere una "filigrana" di rumore globale identica su tutto il set di addestramento. All'interno della singola immagine, però, la casualità o la ripetitività dipendono dalla natura della corruzione:

#### 1. Gaussian Noise (Rumore Gaussiano Additivo)
- **Come funziona**: Modifica il valore naturale dei pixel aggiungendo disturbi estratti da una distribuzione normale (Gaussiana) descritta da una deviazione standard ($\sigma$). Con $\sigma=100$ il rumore è ritenuto molto gravoso.
- **Risultato visivo**: L'immagine appare granulosa e "smerigliata", simulando il disturbo introdotto dai sensori delle fotocamere digitali ad alti ISO, in condizioni di scarsa illuminazione (es. noise termico o fotonico).
- **Tipo di pattern (Casuale)**: **Completamente casuale** a livello spaziale. Il disturbo non presenta alcun pattern ripetitivo.

#### 2. Salt & Pepper (Rumore Sale e Pepe)
- **Come funziona**: Va a sostituire integralmente un certo ammontare di pixel (percentuale impostabile) con pixel o del tutto bianchi (sale) o del tutto neri (pepe). La proporzione tra sale e pepe è anch'essa regolabile.
- **Risultato visivo**: Comparsa di punti netti bianchi e neri distribuiti a macchia di leopardo, tipici dei pixel "bruciati" nei vecchi sensori ottici o delle perdite isolate di pacchetti nella trasmissione radiotelevisiva o digitale analogica.
- **Tipo di pattern (Casuale)**: **Puramente casuale**. La selezione dei pixel corrotti avviene in posizioni pseudocasuali non collegate.

#### 3. Quantizzazione + Dithering (Riduzione della profondità colore)
- **Come funziona**: La quantizzazione riduce la profondità di bit del colore (da 8-bit, 16 milioni di colori, a valori estremi come 2-bit, ecc.). Questa riduzione crea degli stacchi netti o "fasce" di colore compatto (color banding). Per ingannare l'occhio e ridurre lo shock visivo del banding, si applica il *Dithering*, che sgrana i bordi mescolando pixel adiacenti.
- **Risultato visivo**: L'immagine assume un aspetto "rétro", molto simile alle grafiche videoludiche degli anni '80 e '90 oppure alle vecchie GIF, con una palette dei colori impoverita ma con continuità garantita da pattern di pixel.
- **Tipo di pattern (Misto)**: Dipende dal tipo di dithering applicato prima della quantizzazione:
  - **Random Dithering (Casuale)**: Aggiunge del semplice rumore uniforme e isolato prima della scala di quantizzazione. È totalmente imprevedibile e puramente casuale.
  - **Ordered Dithering / Bayer (Fisso e Deterministico)**: Applica l'errore seguendo una mappa a griglia predeterminata (Bayer Matrix, ad es. matricine di blocchi 2x2, 4x4 o 8x8). Simula la retinatura tipografica con un pattern a incrocio (**cross-hatch**) che per definizione è fissato e visibilmente metodico/ripetitivo in tutta l'immagine.
  - **Floyd-Steinberg (Punto-Fisso Dinamico)**: È un algoritmo di error diffusion. Propaga l'errore matematico della quantizzazione ai pixel adiacenti (a destra, in basso). Non usa pattern ripetitivi estranei (**non periodico**), ma produce texture complesse molto organiche che ricalcano in modo deterministico le linee e i bordi dell'immagine originaria.
