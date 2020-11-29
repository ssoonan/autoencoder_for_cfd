
## 한양대학교 2020년 2학기 기계공학종합설계 코드 모음집

### 주제 : Autoencoder를 활용한 CFD 층류 예측

### 파일 구조

```
──data : CFD로 출력된 이미지들
──preprocess.py : csv label을 전처리하여 numpy 배열로 변환
──dataset.py : 데이터들을 최종 numpy 배열로 변환
──train.ipynb : 딥러닝 모델 및 학습, 검증에 해당하는 부분을 노트북으로 나열
──weight.h5 : 학습이 완료된 weight
```