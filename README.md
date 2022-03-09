# Minikeys
### Pilot project searching for BitCoin [minikeys 22 characters](https://en.bitcoin.it/wiki/Mini_private_key_format) on the GPU Cuda.

- The program is under testing, errors are possible.
- Many thanks to [PawelGorny](https://github.com/PawelGorny/MinikeyCuda) in the development of the program. 
- Write a checkpoint every minute to a file fileStatus.txt

Run GPU TEST: ```MinikeyCuda.exe -input test_SkK5VPtmTm3mQKYaJQFRZP.txt -rangeStart SkK5VPtmTm3mQKYaJQ2222```</br>
Run GPU: ```MinikeyCuda.exe -input serie1.txt -rangeStart SkK5VPtmTm3mQKYaJQ2222```

![minikeyS](https://user-images.githubusercontent.com/82582647/157493492-9ba3dbba-0847-4de7-bc10-42f54b49ca3b.jpg)

Run CPU random: ```MinikeyCuda.exe -input -input serie1.txt -random```

![minikeyS](https://user-images.githubusercontent.com/82582647/157494055-f6c787a3-6bd7-4eee-ad7d-21e4d87a8ac3.jpg)


## Building
### Windows
- Microsoft Visual Studio Community 2019
- CUDA version [**10.22**](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork)
## License
- Rotor-Cuda is licensed under GPLv3.

## Donation
- phrutis BTC: bc1qh2mvnf5fujg93mwl8pe688yucaw9sflmwsukz9
- PawelGorny BTC: 34dEiyShGJcnGAg2jWhcoDDRxpennSZxg8
