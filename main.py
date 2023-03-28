from azureml.core import Workspace
ws_other_environment = Workspace.from_config(path="./file-path/ws_config.json")

print('hello world')
