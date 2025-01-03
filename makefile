export PYTHONPATH=$(pwd)

all: venv requirements activate python_path


venv:
	python3 -m venv venv
	bash -c 'source venv/bin/activate && pip install uv'

requirements: venv
	bash -c 'source venv/bin/activate && uv pip install -r requirements.txt'

activate:
	@echo "To activate the virtual environment, run: source venv/bin/activate"

python_path:
	export PYTHONPATH=$(pwd)

.EXPORT_ALL_VARIABLES: