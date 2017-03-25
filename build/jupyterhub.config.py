# Configuration file for jupyterhub.
#  Use with ssl_key
c.JupyterHub.ssl_cert = '/etc/jupyterhub/ssl.crt'

# Path to SSL key file for the public facing interface of the proxy 
#  Use with ssl_cert
c.JupyterHub.ssl_key = '/etc/jupyterhub/ssl.key'

# Whitelist of environment variables for the subprocess to inherit
c.Spawner.env_keep = ['PATH', 'PYTHONPATH', 'CONDA_ROOT', 'CONDA_DEFAULT_ENV', 'VIRTUAL_ENV', 'LANG', 'LC_ALL', 'CUDA_HOME', 'CUDA_ROOT', 'LD_LIBRARY_PATH','LEXICONRN_HOME']

c.NotebookApp.kernel_spec_manager_class = 'environment_kernels.EnvironmentKernelSpecManager'
c.EnvironmentKernelSpecManager.conda_env_dirs=['/opt/anaconda3/envs']
