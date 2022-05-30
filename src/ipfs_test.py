import ipfshttpclient # 須使用 go-ipfs version 在 0.4 ~ 0.8
# 使用環境 ipfs version == 0.7.0, ipfshttpclient version == 0.7.1
client = ipfshttpclient.connect("/ip4/127.0.0.1/tcp/5001")
res = client.add('DCGAN.py')