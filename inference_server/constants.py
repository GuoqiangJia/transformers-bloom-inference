# inference method (args.deployment_framework)
HF_ACCELERATE = "hf_accelerate"
HF_CPU = "hf_cpu"
DS_INFERENCE = "ds_inference"
DS_ZERO = "ds_zero"

redis_url = 'redis://default:redispw@localhost:32768/0'
inference_url = 'http://localhost:7872/generate/'
# GRPC_MAX_MSG_SIZE = 2**30  # 1GB
