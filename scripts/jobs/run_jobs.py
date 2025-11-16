import os

from mls.manager.job.utils import training_job_api_from_profile

if __name__ == "__main__":
    workdir = os.getcwd()
    python_path = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"

    client, extra_options = training_job_api_from_profile("default")

    author_name = "d.tarasov"

    # for embedding_init_method in ["random", "mvnormal"]:
    for embedding_init_method in ["mvnormal"]:
        # for random_seed in [42, 533, 100, 200]:
        for random_seed in [
            42,
            533,
        ]:
            # for random_seed in [100, 200]:
            for max_sequence_length in [32]:
                # for max_sequence_length in [8, 16, 32, 64]:
                result = client.run_job(
                    payload={
                        "script": f" cd {workdir} && {python_path} scripts/activation_distillation.py  --remove_unused_columns False  --num_alignment_layers 1 --loss_type cross_entropy --max_sequence_length {max_sequence_length} --warmup_steps 100 --model_checkpoint HuggingFaceTB/SmolLM2-1.7B --per_device_train_batch_size 1 --max_optimization_steps_per_sample 1000 --learning_rate 0.01  --random_seed {random_seed} --embedding_init_method {embedding_init_method} ",
                        "job_desc": f"CH: compress init={embedding_init_method} seq_len={max_sequence_length} seed={random_seed} #{author_name} #rnd #multimodal @mrsndmn",
                        "env_variables": {
                            "PYTHONPATH": "./src",
                        },
                        "instance_type": "a100.1gpu",
                        "region": extra_options["region"],
                        "type": "binary_exp",
                        "shm_size_class": "medium",
                        "base_image": "cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36",
                        "n_workers": 1,  # Количество воркеров.
                        "processes_per_worker": 1,  # Количество процессов на воркер. Для accelerate нужно запускать 1 процесс на воркер. Для torchrun лучше не заполнять этот параметр. По умолчанию запускается по количеству GPU на одном воркере - это подходит для torchrun.
                    }
                )

                print(embedding_init_method, random_seed, result)
