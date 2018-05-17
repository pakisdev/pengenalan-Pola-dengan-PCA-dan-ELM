test_data = xlsread('data_uji_baru.xlsx');
%memanggil file xlsx 
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
%===============
DataTesting=size(TV.P,2);
temp_TV_T=zeros(OutputNeuron, DataTesting);
    for i = 1:DataTesting
        for j = 1:size(label,2)
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
TV.T=temp_TV_T*2-1;
%===============
MulaiUji=cputime;
tempH_test=InputWeight*TV.P;
%===============
ind=ones(1,DataTesting);
BiasMatrix=HiddenBias(:,ind);         
tempH_test=tempH_test + BiasMatrix;
%====sigmoid===========
H_test = 1 ./ (1 + exp(-tempH_test));
%===============
TY=(H_test' * OutputWeight)';                  
SelesaiUji=cputime;
WaktuUji=SelesaiUji-MulaiUji
%===============
Kesalahanklasifikasi_uji = 0
 for i = 1 : size(TV.T, 2)
        [x, Hasildiharapkan]=max(TV.T(:,i));
        [x, Hasilyangdihasilkan]=max(TY(:,i));
       Hasil_uji(i)=label(Hasilyangdihasilkan); 
         %===Hasil Pengujian=======
        if Hasilyangdihasilkan~=Hasildiharapkan
            Kesalahanklasifikasi_uji=Kesalahanklasifikasi_uji+1;
        end
    end
    Akurasi_Pengujian=1-Kesalahanklasifikasi_uji/DataTesting  
 save('Pengujian');
     %===P=======
 NilaiBenar = 20 - Kesalahanklasifikasi_uji
 Presentase_uji = NilaiBenar / 20 *100
