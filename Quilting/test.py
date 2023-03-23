import os

folder_path = '../images/test_images'

for filename in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, filename)):
        output = filename.split('.')
        command = ('python main.py --image_path ../images/test_images/' + filename + ' -s 2 -b 60 -p 0 -f results/results2/' + output[0] + '_synth.png')
        os.system(command)
        command = ('python main.py --image_path ../images/test_images/' + filename + ' -s 2 -b 60 -p 0 -f results/results3/' + output[0] + '_synth.png')
        os.system(command)
        command = ('python main.py --image_path ../images/test_images/' + filename + ' -s 2 -b 60 -p 0 -f results/results4/' + output[0] + '_synth.png')
        os.system(command)
        command = ('python main.py --image_path ../images/test_images/' + filename + ' -s 2 -b 60 -p 0 -f results/results5/' + output[0] + '_synth.png')
        os.system(command)
