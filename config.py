import os

USER = os.environ['USER']
#USER = 'eyankovsky'
#USER = 'bachman'
project_name = 'oae-dor-global-efficiency'

account = 'P93300612'  #ENTER HERE

dir_project_root = f'/glade/work/{USER}/{project_name}'
os.makedirs(dir_project_root, exist_ok=True)

dir_data = f'{dir_project_root}/data'
os.makedirs(dir_data, exist_ok=True)

dir_scratch = f'/glade/scratch/bachman'


#### Added on May 28, 2023, by Mengyang
# if saving cesm-cases and archive to sracth
#dir_project_root_scratch = f'/glade/scratch/{USER}/{project_name}'
#os.makedirs(dir_project_root_scratch, exist_ok=True)

# if reading forcing from sracth
#dir_data_scratch = f'{dir_project_root_scratch}/data'
#os.makedirs(dir_data_scratch, exist_ok=True)


#dir_scratch = f'/glade/scratch/eyankovsky/'
#dir_project_root_scratch = f'/glade/scratch/eyankovsky/{project_name}'
#os.makedirs(dir_project_root_scratch, exist_ok=True)

