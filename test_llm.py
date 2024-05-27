from llm_client import AlpacaLLM
prompt = """
### Instruction:
your job is to answer question based on a given pieces of information. all you have to do is answer the question. not all of the information provided maybe relevant to the question. each pieces of information will be separated by '---'

### Example:
Question: Apa manfaat olahraga teratur bagi kesehatan kardiovaskular?
---

Penelitian yang dipublikasikan dalam Journal of the American Heart Association menunjukkan bahwa olahraga teratur dapat mengurangi risiko penyakit jantung koroner hingga 30%. Aktivitas fisik membantu memperkuat otot jantung, meningkatkan sirkulasi darah, dan menurunkan tekanan darah.

---

Meskipun olahraga memiliki banyak manfaat, penting untuk dilakukan dengan benar untuk menghindari cedera. Pemanasan sebelum berolahraga dan pendinginan setelahnya sangat dianjurkan. Selain itu, jenis olahraga yang dipilih harus sesuai dengan kondisi fisik individu untuk menghindari risiko yang tidak diinginkan.

---

Menurut sebuah studi dari Mayo Clinic, orang yang rutin berolahraga memiliki kadar kolesterol yang lebih baik dan cenderung memiliki berat badan yang lebih sehat. Olahraga juga dapat meningkatkan sensitivitas insulin dan membantu mengatur kadar gula darah, yang merupakan faktor penting dalam menjaga kesehatan jantung.
---

Answer:
Regular physical exercise has several benefits for cardiovascular health. Firstly, it can reduce the risk of coronary heart disease by up to 30%, as it strengthens the heart muscles, improves blood circulation, and lowers blood pressure. Secondly, individuals who exercise regularly tend to have better cholesterol levels and a healthier weight, which are crucial for heart health. Additionally, regular exercise can increase insulin sensitivity and help regulate blood sugar levels, further contributing to cardiovascular well-being.

### Another example:
Question: Apa manfaat diet kaya serat untuk kesehatan pencernaan?
---

Diet kaya serat diketahui dapat mencegah sembelit dengan meningkatkan volume dan kelembutan tinja, membuatnya lebih mudah untuk dikeluarkan. Serat juga membantu menjaga kesehatan usus dengan mempromosikan pertumbuhan bakteri baik di dalam sistem pencernaan.

---

Makanan tinggi serat seperti buah-buahan, sayuran, dan biji-bijian utuh tidak hanya baik untuk pencernaan, tetapi juga dapat membantu mengontrol kadar gula darah dan menurunkan kadar kolesterol. Serat larut dalam makanan ini membantu memperlambat penyerapan gula dan mengikat kolesterol di usus.

---

Beberapa penelitian menunjukkan bahwa diet tinggi serat dapat mengurangi risiko kanker kolorektal. Serat membantu mempercepat pembuangan zat-zat karsinogenik dari usus besar, mengurangi waktu paparan sel-sel usus terhadap bahan-bahan berbahaya.
---

Jawaban:
A diet rich in fiber has multiple benefits for digestive health. It can prevent constipation by increasing stool bulk and softness, making it easier to pass. Fiber also promotes gut health by encouraging the growth of beneficial bacteria in the digestive system. Additionally, high-fiber foods such as fruits, vegetables, and whole grains can help control blood sugar levels and lower cholesterol. Soluble fiber in these foods slows sugar absorption and binds cholesterol in the intestines. Furthermore, a high-fiber diet can reduce the risk of colorectal cancer by speeding up the removal of carcinogenic substances from the colon, thereby reducing the exposure time of colon cells to harmful materials.

### Input
Question: Apa penyebab kerusakan ginjal pada anak?:

---

abnormalitas struktur atau fungsi ginjal dengan atau tanpa penurunan LFG, yang bermanifestasi sebagai satu atau lebih gejala :i).Abnormalitas komposisi urin ii).Abnormalitas pemeriksaan pencitraan iii).Abnormalitas biopsi ginjal 
Berdasarkan nilai LFG yang tersisa penyakit ginjal 
kronis diklasifikasikan dalam 5 stadium, seperti dalam tabel berikut:
Etiologi
Pada anak, penyakit ginjal kronis dapat disebabkan 
penyakit kongenital, didapat, genetik, atau metabolik. Penyebab yang mendasari berkaitan erat dengan usia pasien saat penyakit ginjal kronis pertama terdeteksi. Penyakit ginjal kronis pada anak yang berusia kurang dari 5 tahun biasa disebabkan abnormalitas kongenital seperti hipoplasia atau displasia ginjal, dan/atau uropati obstruktif. Penyebab lain adalah sindrom nefrotik kongenital, sindrom prune belly , nekrosis korteks, 
glomerulosklerosis fokal segmental, penyakit ginjal polikistik, trombosis vena renalis, dan sindrom hemolitik uremik.Setelah usia 5 tahun, penyakit-penyakit didapat (berbagai bentuk glomerulonefritis termasuk lupus nefritis) lebih mendominasi. Terdapat 
beberapa faktor yang menyebabkan kerusakan
berkelanjutan pada penyakit ginjal kronis, yaitu 
glomerulosklerosis, pembentukan fibrosis tubulointerstisial, proteinuria, dan sklerosis vaskular.
Diagnosis 
Penderita penyakit ginjal kronis stadium 1-3 (LFG>30

---

ABSTRAK
Penyakit ginjal kronik (PGK) merupakan masalah kesehatan serius pada anak dengan morbiditas dan mortalitas yang makin meningkat serta 
menimbulkan masalah sosial ekonomi yang signifikan. Penyebab utama PGK pada anak adalah anomali kongenital ginjal dan saluran kemih, 
diikuti nefropati herediter dan glomerulonefritis. Indikasi absolut untuk memulai dialisis pada anak meliputi anuria, gangguan elektrolit berat, 
gangguan neurologis pada gagal ginjal, perikarditis, diatesis perdarahan, mual berulang, gejala uremia, volume berlebihan, atau kegagalan 
pertumbuhan meskipun terapi medis sudah tepat, dan hipertensi. Mengoptimalkan status gizi dengan dukungan ahli diet adalah elemen dasar 
perawatan anak dengan penyakit ginjal kronik stadium 5. Peran ahli diet adalah meminimalkan gejala uremia, mencegah komplikasi penyakit 
tulang akibat gangguan ginjal (renal bone disease), dan mengoptimalkan pertumbuhan anak.
Kata kunci: Anak, hemodialisis, indikasi, penyakit ginjal kronik

---

Langkah pertama dalam proses deteksi dini PGK adalah dengan 
pemeriksaan urinalisis (mencari proteinuria, hematuria, dan cellular cast ). 
Evaluasi lebih lanjut termasuk beratnya proteinuria, USG ginjal, rujukan 
ke dokter spesialis penyakit dalam/spesialis anak, rujukan ke dokter 
spesialis penyakit dalam konsultan ginjal hipertensi/spesialis anak 
subspesialis nefrologi, dan biopsi ginjal. Rujukan ke ahli nefrologi terutama diindikasikan pada eGFR <30ml/menit/1,73m
2, penurunan 
fungsi ginjal yang cepat, peningkatan rasio albumin-kreatinin (>300 mg/g), atau didapatkan eritrosit cast  di urin.  Rekomendasi 1:  Kerusakan pada ginjal dapat diakibatkan oleh beberapa keadaan.  Langkah pertama dalam proses detekpemeriksaan urinalisis. Evaluasi lebih lanjut termasuk penurunan eGFR, beratnya proteinuria, USG ginjal, rujukan ke dokter spesialis penyakit dalam/spesialis anak, rujukan ke dokter spesialis penyakit dalam konsultan ginjal hipertensi /spesialis anak subspesialis nefrologi, 
dan tindakan biopsi ginjal. B. Etiologi Penyakit Ginjal Kronik
Etiologi PGK adalah diabetes melitus, hipertensi, glomerulonefritis 
primer, nefritis tubulointersisial kronik, penyakit k
ista ginjal herediter, 
glomerulonephritis sekunder atau vaskulitis dan neoplasma. Pada anak, 
penyebab primer PGTA yang tersering adalah kelainan kongenital ginjal dan saluran kemih (seperti obstruksi uropati, refluks vesikoureter, 
displasia ginjal) sebesar 37%; diikuti kelainan glomerular seperti sindrom

---

"""

model = AlpacaLLM()
# instruct = model.invoke(instruction)
# print(instruct)
response_text = model.invoke(prompt)
print(response_text)