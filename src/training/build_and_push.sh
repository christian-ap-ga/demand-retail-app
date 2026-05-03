#!/usr/bin/env bash
# Construye el Docker image localmente y lo sube a ECR para que SageMaker pueda usarlo.
#
# El script realiza los siguientes pasos:
#   1. Obtiene el account ID y la region desde las credenciales AWS activas
#   2. Crea el repositorio en ECR si no existe
#   3. Autentica Docker contra ECR
#   4. Construye el image, lo tagea con el URI completo de ECR y lo sube
#
# Uso:
#   bash build_and_push.sh <image-name>
#
# Parámetros:
#   image-name  Nombre del image local y del repositorio en ECR
#
# Ejemplo:
#   bash build_and_push.sh decision-trees
#
# Nota (SageMaker Studio): el daemon de Docker en SageMaker Studio solo permite
# la red 'sagemaker' durante el build. Por eso se usa --network sagemaker en
# el comando docker build.

# Nombre del image — también se usará como nombre del repositorio en ECR.
image=$1

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

# Asegurar que los scripts de entrenamiento e inferencia sean ejecutables.
chmod +x decision_trees/train
chmod +x decision_trees/serve

# Obtener el account ID asociado a las credenciales IAM activas.
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

# Obtener la region de la configuración AWS. Si no está definida, usar us-east-1.
region=$(aws configure get region)
region=${region:-us-east-1}

# URI completo del image en ECR: <account>.dkr.ecr.<region>.amazonaws.com/<image>:latest
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# Crear el repositorio en ECR si no existe.
# La salida se descarta; solo interesa el exit code.
aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Autenticar Docker contra ECR usando el token temporal de la sesión AWS.
aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${account}".dkr.ecr."${region}".amazonaws.com

# Construir el image localmente.
# --network sagemaker es requerido en SageMaker Studio para permitir acceso a red durante el build.
docker build --network sagemaker -t ${image} .

# Tagear el image con el URI completo de ECR y subirlo.
docker tag ${image} ${fullname}
docker push ${fullname}
