import psutil
import wandb

def print_file_info():
    proc = psutil.Process()
    print('Num open files: %s' % len(proc.open_files()))
    for filename in proc.open_files():
        print('\t%s' % filename.path)

print('Before any WANDB stuff')
print_file_info()
run = wandb.init(project='test', name='test', reinit=True, resume='allow')
run_id = run.id
run.finish()

print('After creating run and getting run ID')
print_file_info()
run = wandb.init(id=run_id, project='test', name='test', reinit=True, resume='allow')
run.finish()

print('After accessing run again')
print_file_info()

test_file = open('test_file.txt', 'w')
print('After creating a normal file pointer')
print_file_info()

test_file.close()
print('After closing that file')
print_file_info()