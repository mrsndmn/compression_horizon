import os

from mls.manager.job.utils import training_job_api_from_profile

if __name__ == "__main__":
    workdir = os.getcwd()
    python_path = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"

    client, extra_options = training_job_api_from_profile("default")

    author_name = "d.tarasov"

    checkpoints = [
        # ["random_128", "ch_cross_entropy_init_random_fmivdb", "ch_cross_entropy_init_random_qjabzg"],
        # ["mvnormal_128", "ch_cross_entropy_init_mvnormal_qghjjl", "ch_cross_entropy_init_mvnormal_kgkutv"],
        #
        # ["mvnormal_4", "ch_cross_entropy_init_mvnormal_seq_len_4_dldqpt", "ch_cross_entropy_init_mvnormal_seq_len_4_lvectf"],
        # ["mvnormal_4", "ch_cross_entropy_init_mvnormal_seq_len_4_dldqpt", "ch_cross_entropy_init_mvnormal_seq_len_4_jygdfi"],
        # ["mvnormal_4", "ch_cross_entropy_init_mvnormal_seq_len_4_dldqpt", "ch_cross_entropy_init_mvnormal_seq_len_4_vxzytl"],
        # ["mvnormal_4", "ch_cross_entropy_init_mvnormal_seq_len_4_lvectf", "ch_cross_entropy_init_mvnormal_seq_len_4_jygdfi"],
        # ["mvnormal_4", "ch_cross_entropy_init_mvnormal_seq_len_4_lvectf", "ch_cross_entropy_init_mvnormal_seq_len_4_vxzytl"],
        # ["mvnormal_4", "ch_cross_entropy_init_mvnormal_seq_len_4_jygdfi", "ch_cross_entropy_init_mvnormal_seq_len_4_vxzytl"],
        #
        ["mvnormal_8", "ch_cross_entropy_init_mvnormal_seq_len_8_vmvyuv", "ch_cross_entropy_init_mvnormal_seq_len_8_wfehxf"],
        ["mvnormal_8", "ch_cross_entropy_init_mvnormal_seq_len_8_vmvyuv", "ch_cross_entropy_init_mvnormal_seq_len_8_qbiwve"],
        ["mvnormal_8", "ch_cross_entropy_init_mvnormal_seq_len_8_vmvyuv", "ch_cross_entropy_init_mvnormal_seq_len_8_igsrme"],
        ["mvnormal_8", "ch_cross_entropy_init_mvnormal_seq_len_8_wfehxf", "ch_cross_entropy_init_mvnormal_seq_len_8_qbiwve"],
        ["mvnormal_8", "ch_cross_entropy_init_mvnormal_seq_len_8_wfehxf", "ch_cross_entropy_init_mvnormal_seq_len_8_igsrme"],
        ["mvnormal_8", "ch_cross_entropy_init_mvnormal_seq_len_8_qbiwve", "ch_cross_entropy_init_mvnormal_seq_len_8_igsrme"],
        #
        # ["mvnormal_16", "ch_cross_entropy_init_mvnormal_seq_len_16_egczqf", "ch_cross_entropy_init_mvnormal_seq_len_16_cxaxop"],
        # ["mvnormal_16", "ch_cross_entropy_init_mvnormal_seq_len_16_egczqf", "ch_cross_entropy_init_mvnormal_seq_len_16_bxcseh"],
        # ["mvnormal_16", "ch_cross_entropy_init_mvnormal_seq_len_16_egczqf", "ch_cross_entropy_init_mvnormal_seq_len_16_wrghnk"],
        # ["mvnormal_16", "ch_cross_entropy_init_mvnormal_seq_len_16_cxaxop", "ch_cross_entropy_init_mvnormal_seq_len_16_bxcseh"],
        # ["mvnormal_16", "ch_cross_entropy_init_mvnormal_seq_len_16_cxaxop", "ch_cross_entropy_init_mvnormal_seq_len_16_wrghnk"],
        # ["mvnormal_16", "ch_cross_entropy_init_mvnormal_seq_len_16_bxcseh", "ch_cross_entropy_init_mvnormal_seq_len_16_wrghnk"],
        #
        # ["mvnormal_32", "ch_cross_entropy_init_mvnormal_seq_len_32_vgvpdr", "ch_cross_entropy_init_mvnormal_seq_len_32_gpvzsk"],
        # ["mvnormal_32", "ch_cross_entropy_init_mvnormal_seq_len_32_vgvpdr", "ch_cross_entropy_init_mvnormal_seq_len_32_qngnds"],
        # ["mvnormal_32", "ch_cross_entropy_init_mvnormal_seq_len_32_vgvpdr", "ch_cross_entropy_init_mvnormal_seq_len_32_jrkkga"],
        # ["mvnormal_32", "ch_cross_entropy_init_mvnormal_seq_len_32_gpvzsk", "ch_cross_entropy_init_mvnormal_seq_len_32_qngnds"],
        # ["mvnormal_32", "ch_cross_entropy_init_mvnormal_seq_len_32_gpvzsk", "ch_cross_entropy_init_mvnormal_seq_len_32_jrkkga"],
        # ["mvnormal_32", "ch_cross_entropy_init_mvnormal_seq_len_32_qngnds", "ch_cross_entropy_init_mvnormal_seq_len_32_jrkkga"],
        #
        # ["mvnormal_64", "ch_cross_entropy_init_mvnormal_seq_len_64_pcuxkf", "ch_cross_entropy_init_mvnormal_seq_len_64_emaoab"],
    ]

    for exp_type, checkpoint1, checkpoint2 in checkpoints:
        out_dir_name = f"{exp_type}_{checkpoint1}_{checkpoint2}"

        bezier_steps = 5000

        result = client.run_job(
            payload={
                "script": f" cd {workdir} && {python_path} scripts/interpolation.py --dataset_path1 artifacts/experiments/{checkpoint1}/compressed_prefixes --dataset_path2 artifacts/experiments/{checkpoint2}/compressed_prefixes --bezier_steps {bezier_steps} --bezier_batch_t 100 --bezier_lr 0.1 --bezier_weight_decay 0.0 --bezier_order 2 --output_dir artifacts/interpolations/{out_dir_name}",
                "job_desc": f"CH: interpolate {checkpoint1} {checkpoint2} #{author_name} #multimodal @mrsndmn",
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
        print(checkpoint1, checkpoint2, result)
