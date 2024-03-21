from deepjanus_bng.beamng_config import BeamNGConfig
from deepjanus_mnist.mnist_config import MNISTConfig
import json

REGIONS = [0.8, 0.2, 0.5, 0.6]


def generate_run_configuration(base_cfg, cfg_file_path, cmd, script_preamble, script_file):
    run_script = script_preamble
    for i in range(1, 11):
        for region in REGIONS:
            region_cfg = base_cfg.clone()
            region_cfg.PROBABILITY_THRESHOLD = region
            region_name = str(int(region * 100))
            exp_name = region_name + "_" + str(i)
            region_cfg.EXPERIMENT_NAME = exp_name
            region_cfg.SEED_FOLDER = region_name

            with open(cfg_file_path.format(exp_name), "w") as f:
                f.write(json.dumps(region_cfg.__dict__))

            run_script += cmd.format(exp_name) + "\n"

    with open(script_file, "w") as f:
        f.write(run_script)


def setup_mnist_experiments():
    mnist_cfg = MNISTConfig("../DeepJanus-MNIST")
    generate_run_configuration(mnist_cfg,
                               "../DeepJanus-MNIST/data/experiments/{}.json",
                               "python deepjanus_mnist.py -c data/experiments/{}.json",
                               "#!/bin/bash\nset -e\n",
                               "../DeepJanus-MNIST/run_all.sh")


def setup_beamng_experiments():
    beamng_cfg = BeamNGConfig("../DeepJanus-BNG")
    generate_run_configuration(beamng_cfg,
                               "../DeepJanus-BNG/data/experiments/{}.json",
                               "python deepjanus_beamng.py -c data/experiments/{}.json",
                               "",
                               "../DeepJanus-BNG/run_all.bat")


if __name__ == "__main__":
    setup_beamng_experiments()
    setup_mnist_experiments()
