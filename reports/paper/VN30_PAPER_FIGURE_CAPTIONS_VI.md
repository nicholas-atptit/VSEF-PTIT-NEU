# VN30 Paper Figure Captions VI

## Hinh 1. Tom tat so sanh benchmark giua cac phuong phap va ky han

- Data: Su dung cac CSV tong hop cho ung vien VN30 co phieu, benchmark chi so, va panel ghep co phieu+chi so.
- Shows: So sanh do chinh xac chieu bien dong va baseline.
- Interpretation: Ket qua stock-only la bang chung chinh; chi so va panel ghep la boi canh.
- Caveats: Panel ghep chi co tong hop va khong phai claim thanh cong.

## Hinh 2. Ung vien tot nhat: thuc te so voi du bao theo thoi gian

- Data: Su dung du lieu row-level tai lap L2 Logistic h40.
- Shows: The hien huong thuc te, huong du bao, va rolling accuracy.
- Interpretation: Ho tro dien giai tin hieu dinh huong co y nghia quanh muc tren 60%.
- Caveats: Can cong bo validation-final gap va do on dinh theo thoi gian con lan lon.

## Hinh 3. So sanh du bao-thuc te tren cac ma dai dien

- Data: Su dung row-level predictions cho co phieu dai dien va chi so.
- Shows: So sanh cac phuong phap cot loi theo tung instrument.
- Interpretation: Cho thay hieu qua khac nhau theo model va instrument.
- Caveats: Khong co row-level predictions cho stacking.

## Hinh 4. Phan phoi do chinh xac theo instrument

- Data: Su dung slice theo ticker va benchmark chi so.
- Shows: Xep hang do chinh xac co phieu va chi so rieng.
- Interpretation: Do chinh xac tong hop khong dong deu.
- Caveats: Hai panel den tu cac artifact hop le khac nhau.

## Hinh 5. Do on dinh cua ung vien duoc chon

- Data: Su dung rolling 250/500/1000, thang, va quy.
- Shows: The hien accuracy theo cua so rolling va ky lich.
- Interpretation: Tin hieu tong hop di kem bat on theo thoi gian.
- Caveats: Rolling la theo so dong cua panel.

## Hinh 6. Overlay model voi thuc te

- Data: Su dung mot co phieu va mot chi so dai dien.
- Shows: Dat huong thuc te va du bao len cung timeline.
- Interpretation: Lam ro cac giai doan dong thuan va bat dong.
- Caveats: Chi so khong co L2 Logistic va stacking row-level.
