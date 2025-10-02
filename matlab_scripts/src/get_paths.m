function paths = get_paths()
addpath(genpath("../yamlmatlab")) % in matlab scripts

% add the yamlmatlab folder to your MATLAB path first
config = yaml.ReadYaml('../../config.yaml'); % in the general folder

% get environment variable (like Python's os.getenv)
env = getenv('MY_ENV');
if isempty(env)
    warning("The environment is empty");  
end

% now access the config just like in Python
paths = config.(env).paths;

