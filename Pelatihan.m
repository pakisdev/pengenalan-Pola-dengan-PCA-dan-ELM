train_data = xlsread('data_latih_baru.xlsx');
T=train_data(:,1)';
%T transpose merukan nilai kolom pada baris pertama pada  train data 
P=train_data(:,2:size(train_data,2))';
%P Transpose merupakan nilai dari bari kedua dan seterusnya pada train data 
TrainingData= size(P,2);
%jumlah data latih
InputNeuron= size(P,1);
%jumlah Input Nueron
%====menentukan label======
sorted_target=sort(T,2);
    label=zeros(1,1);                              
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:TrainingData
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    OutputNeuron=number_class;
    %====Inisialisasi bobot dan bias dengan bilangan acak yang kecil, tergantung fungsi aktivasi yang digunakan.======
%25 hidden neuron==========
temp_T=zeros(50, TrainingData);
    for i = 1:TrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1; 
%==========
MulaiPelatihan=cputime;
%25 hidden neuron==========
InputWeight=rand(50,InputNeuron)*2-1;
%Input weight (w_i) diambil secarak acak menggunakan fungsi rand (20 Jumlah hidden neuron, InputNeuron) dikali 2-1
HiddenBias= rand(50,1);
%Hiddenbias (b_i) diambil secara acak menggunakan fungsi ran 
%dengan menentukan jumlah Hidden neuron kemudian di ambil hanya pada baris pertama
tempH=InputWeight*P;
%tempH merupakan hasil perkalian dari input weight dengan P 
ind=ones(1,TrainingData);
BiasMatrix=HiddenBias(:,ind);
tempH=tempH+BiasMatrix;
%===Aktivasi sigmoid=======
H=1./(1+exp(-tempH));
%fungsi aktivasi sigmoid
OutputWeight=pinv(H') * T';
SelesaiPelatihan=cputime;
WaktuPelatihan=SelesaiPelatihan-MulaiPelatihan 
%Outputweight
Y=(H' *OutputWeight)';
%===akurasi=======
Kesalahanklasifikasi_pelatihan=0;
    for i = 1 : size(T, 2)
        [x, Hasildiharapkan]=max(T(:,i));
        [x, Hasilyangdihasilkan]=max(Y(:,i));
        Hasil_latih(i)=label(Hasilyangdihasilkan);
        %===Hasil Pelatihan=======
        if Hasilyangdihasilkan~=Hasildiharapkan
            Kesalahanklasifikasi_pelatihan=Kesalahanklasifikasi_pelatihan+1;
        end
    end
    Akurasi_Pelatihan=1-Kesalahanklasifikasi_pelatihan/TrainingData
    save('Pelatihan');
NilaiBenarP = 80 - Kesalahanklasifikasi_pelatihan
Presentase_pelatihan = NilaiBenarP / 80 *100
