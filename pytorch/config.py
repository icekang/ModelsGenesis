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
    train_fold=['03-008', '708-002', '101-051', '03-009', '706-004', '402-006', '703-002', '706-009', '704-001', '02-005', '402-009', '106-019', '101-032', '101-002', '106-016', '706-014', '702-003', '101-038', '04-010', '701-007', '101-028', '06-001', '101-057', '401-004', '703-001', '101-042', '101-012', '403-003', '705-002', '03-007', '101-015', '707-005', '101-006', '701-004', '101-009', '131-011', '135-003', '101-054', '03-003', '402-008', '701-002', '101-047', '101-020', '101-019', '702-005', '03-005', '101-036', '139-005', '403-006', '706-003', '706-011', '03-006', '04-006', '707-002', '03-001', '05-001', '101-026', '401-003', '705-005', '03-004', '708-007', '708-001', '706-001', '101-055', '101-050', '702-007', '101-040', '702-008', '106-017', '708-006', '403-002', '705-006', '701-015', '101-033', '106-009', '701-014', '101-044', '701-011', '706-012', '106-002', '401-008', '101-014', '708-003', '106-010', '109-003', '706-010', '101-024', '101-010', '705-001', '707-004', '02-008', '101-053', '05-011', '02-013', '06-003', '704-003', '703-004', '706-007', '701-008', '707-006', '401-005', '706-013', '708-008', '05-009', '401-002', '704-006', '04-008', '702-002', '704-002']
    valid_fold=['704-005', '704-004', '04-001', '101-007', '707-001', '131-013', '131-003', '706-006', '708-005', '701-006', '701-005', '101-039', '139-004', '131-016', '101-013', '101-035', '101-008', '05-005', '708-004', '101-058', '707-003', '03-002', '705-004', '101-022', '701-013', '703-005', '04-014', '05-002']
    test_fold=[]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 96
    input_rows = 64
    input_cols = 64 
    input_deps = 32
    nb_class = 1
    
    # model pre-training
    verbose = 1
    weights = None
    batch_size = 12
    optimizer = "sgd"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = 10000
    patience = 100
    lr = 1

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    # logs
    model_path = "/storage_bizon/naravich/ModelGenesisPretrainingV2"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    # wandb
    wandb_project_name = "Genesis_Pretraing"
    wandb_run_name = None
    wandb_run_id = None # for a new run, it will be auto generated, otherwise, specify the run id to resume

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def to_dict(self):
        return {a: getattr(self, a) for a in dir(self) if not a.startswith("__") and not callable(getattr(self, a))}