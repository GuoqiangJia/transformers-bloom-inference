gen-proto:
	mkdir -p inference_server/model_handler/grpc_utils/pb

	python -m grpc_tools.protoc -Iinference_server/model_handler/grpc_utils/proto --python_out=inference_server/model_handler/grpc_utils/pb --grpc_python_out=inference_server/model_handler/grpc_utils/pb inference_server/model_handler/grpc_utils/proto/generation.proto

	find inference_server/model_handler/grpc_utils/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;

	touch inference_server/model_handler/grpc_utils/__init__.py
	touch inference_server/model_handler/grpc_utils/pb/__init__.py

	rm -rf inference_server/model_handler/grpc_utils/pb/*.py-e

ui:
	python -m ui --ui_host 0.0.0.0 --ui_port 7863 --generation_backend_host 127.0.0.1 --generation_backend_port 7862 &

# ------------------------- DS inference -------------------------
bloom-176b:
	make ui

	TOKENIZERS_PARALLELISM=false \
	MODEL_NAME=bigscience/bloom \
	MODEL_CLASS=AutoModelForCausalLM \
	DEPLOYMENT_FRAMEWORK=ds_inference \
	DTYPE=fp16 \
	MAX_INPUT_LENGTH=2048 \
	MAX_BATCH_SIZE=4 \
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
	gunicorn -t 0 -w 1 -b 127.0.0.1:5000 inference_server.server:app --access-logfile - --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s'

# loads faster than the above one
microsoft-bloom-176b:
	make ui

	TOKENIZERS_PARALLELISM=false \
	MODEL_NAME=microsoft/bloom-deepspeed-inference-fp16 \
	MODEL_CLASS=AutoModelForCausalLM \
	DEPLOYMENT_FRAMEWORK=ds_inference \
	DTYPE=fp16 \
	MAX_INPUT_LENGTH=2048 \
	MAX_BATCH_SIZE=4 \
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
	gunicorn -t 0 -w 1 -b 127.0.0.1:5000 inference_server.server:app --access-logfile - --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s'

bloomz-176b:
	make ui

	TOKENIZERS_PARALLELISM=false \
	MODEL_NAME=bigscience/bloomz \
	MODEL_CLASS=AutoModelForCausalLM \
	DEPLOYMENT_FRAMEWORK=ds_inference \
	DTYPE=fp16 \
	MAX_INPUT_LENGTH=2048 \
	MAX_BATCH_SIZE=4 \
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
	gunicorn -t 0 -w 1 -b 127.0.0.1:5000 inference_server.server:app --access-logfile - --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s'

bloom-176b-int8:
	make ui

	TOKENIZERS_PARALLELISM=false \
	MODEL_NAME=microsoft/bloom-deepspeed-inference-int8 \
	MODEL_CLASS=AutoModelForCausalLM \
	DEPLOYMENT_FRAMEWORK=ds_inference \
	DTYPE=int8 \
	MAX_INPUT_LENGTH=2048 \
	MAX_BATCH_SIZE=4 \
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
	gunicorn -t 0 -w 1 -b 127.0.0.1:5000 inference_server.server:app --access-logfile - --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s'

bloom-mt0-xxl-mt:
	make ui

	TOKENIZERS_PARALLELISM=false \
	MODEL_NAME=bigscience/mt0-xxl-mt \
	MODEL_CLASS=AutoModelForSeq2SeqLM \
	DEPLOYMENT_FRAMEWORK=hf_accelerate \
	DTYPE=bf16 \
	MAX_INPUT_LENGTH=2048 \
	MAX_BATCH_SIZE=4 \
	CUDA_VISIBLE_DEVICES=1,2,3 \
	gunicorn -t 0 -w 1 -b 0.0.0.0:7862 inference_server.server:app --access-logfile /src/logs/access.log --error-logfile /src/logs/error.log --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s'

# ------------------------- HF accelerate -------------------------
bloom-560m:
	make ui

	TOKENIZERS_PARALLELISM=false \
	MODEL_NAME="/src/bloom-560m/"
	MODEL_CLASS=AutoModelForCausalLM \
	DEPLOYMENT_FRAMEWORK=hf_accelerate \
	DTYPE=bf16 \
	MAX_INPUT_LENGTH=2048 \
	MAX_BATCH_SIZE=32 \
	CUDA_VISIBLE_DEVICES=1,2,3 \
	gunicorn -t 0 -w 1 -b 0.0.0.0:7862 inference_server.server:app --access-logfile - --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s'

# ------------------------- HF CPU -------------------------
bloom-560m-cpu:
	make ui

	MODEL_NAME="/src/bloom-560m/" \
	MODEL_CLASS=AutoModelForCausalLM \
	DEPLOYMENT_FRAMEWORK=hf_cpu \
	DTYPE=fp32 \
	MAX_INPUT_LENGTH=2048 \
	MAX_BATCH_SIZE=32 \
	gunicorn -t 0 -w 1 -b 0.0.0.0:7862 inference_server.server:app  --access-logfile /src/logs/access.log --error-logfile /src/logs/error.log --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s'