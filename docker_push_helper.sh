docker push niladrimlops.azurecr.io/dependency_digits
az acr build --image dependency_digits --registry niladrimlops --file ./docker/DependencyDockerfile .

docker push niladrimlops.azurecr.io/digits:v1
az acr build --image digits:v1 --registry niladrimlops --file ./docker/Dockerfile .