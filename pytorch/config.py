import os
import shutil

class models_genesis_config:
    # general
    seed = 0

    model = "Unet3D"
    suffix = "genesis_oct"
    exp_name = model + "-" + suffix
    
    # data
    data = "/storage_bizon/naravich/Unlabeled_OCT_cubes/"
    train_fold=["02-005","02-008","02-013","03-001","03-002","03-003","03-004","03-005","03-006","03-007","03-008","03-009","04-001","04-006","04-008","04-010","04-014","05-001","05-002","05-005","05-009","05-011","06-001","06-003","101-002","101-006","101-007","101-008","101-009","101-010","101-012","101-013","101-014","101-015","101-019","101-020","101-022","101-024","101-026","101-028","101-032","101-033","101-035","101-036","101-038","101-039","101-040","101-042","101-044","101-047","101-050","101-051","101-053","101-054","101-055","101-057","101-058","106-002","106-009","106-010","106-016","106-017","106-019","109-003","131-003","131-011","131-013","131-016","135-003","139-004","139-005","401-002","401-003","401-004","401-005","401-008","402-006","402-008","402-009","403-002","403-003","403-006","701-002","701-004","701-005","701-006","701-007","701-008","701-011","701-013","701-014","701-015","702-002","702-003","702-005","702-007","702-008","703-001","703-002","703-004","703-005","704-001","704-002","704-003","704-004","704-005","704-006","705-001","705-002"]
    valid_fold=["705-004","705-005","705-006","706-001","706-003","706-004","706-006","706-007","706-009","706-010","706-011","706-012","706-013","706-014","707-001","707-002","707-003","707-004","707-005","707-006","708-001","708-002","708-003","708-004","708-005","708-006","708-007","708-008"]
    test_fold=[]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 32
    input_rows = 64
    input_cols = 64 
    input_deps = 32
    nb_class = 1
    
    # model pre-training
    verbose = 1
    weights = "/storage_bizon/naravich/ModelGenesisPretraining/Genesis_Chest_CT.pt"
    batch_size = 12
    optimizer = "sgd"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = 10000
    patience = 50
    lr = 1

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    # logs
    model_path = "/storage_bizon/naravich/ModelGenesisPretraining"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    # wandb
    wandb_project_name = "Genesis_Pretraing"
    wandb_run_name = None
    wandb_run_id = "i9gcmh43" # for a new run, it will be auto generated, otherwise, specify the run id to resume

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def to_dict(self):
        return {a: getattr(self, a) for a in dir(self) if not a.startswith("__") and not callable(getattr(self, a))}