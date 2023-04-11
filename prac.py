import torch

# 모델 가중치 불러오기
weights = torch.load('/home/work/capstone/Final/dancing/AdaCoF_ori/model50.pth', map_location=torch.device('cpu'))

# 모델 파라미터 이름 출력하기
for name in weights:
    print(name)