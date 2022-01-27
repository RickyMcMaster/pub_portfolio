FROM public.ecr.aws/lambda/python:3.8
COPY cats-dogs-v2.tflite .

RUN pip3 install keras_image_helper
RUN pip3 install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl

# COPY dogs-cats-model.tflite .
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]