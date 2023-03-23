import Run_synthesis as rs
from Arg_Parser import get_parser_args 
import glob
import os


max_iter = 200
print_iter = 1

# dataset_folder = '/home/thomas/Downloads/CompImg/flower_beds_256/'
dataset_folder = '/home/thomas/Downloads/CompImg/TestData256'

def runTexture(image_path):
    parser = get_parser_args()
    # name_texture = 'TilesOrnate0158_1_S' # Image in size 256*256
    name_texture = os.path.basename(image_path)
    output_img_name = name_texture + '_Gram'
    parser.set_defaults(img_folder=dataset_folder, verbose=True,max_iter=max_iter,print_iter=print_iter,\
        texture_ref_name=name_texture,loss=['Gram'],output_img_name=output_img_name, MS_Strat='Init', K=3)
    args = parser.parse_args()
    rs.run_synthesis(args)
    print('End of the texture test with Gram Matrices')
    
def runTextureAucorr(image_path):
    parser = get_parser_args()
    # name_texture = 'TilesOrnate0158_1_S' # Image in size 256*256
    name_texture = os.path.basename(image_path)
    output_img_name = name_texture + '_autocorr'
    parser.set_defaults(img_folder=dataset_folder, verbose=True,max_iter=max_iter,print_iter=print_iter,\
        texture_ref_name=name_texture,loss=['autocorr'],output_img_name=output_img_name) 
    args = parser.parse_args()
    rs.run_synthesis(args)      
    print('End of the autocorr texture test')
  
def runTextureMSInit():
    parser = get_parser_args()
    name_texture = 'TexturesCom_BrickSmallBrown0473_1_M_1024' # Image in size 1024*1024
    output_img_name = name_texture + '_Gram_Spectrum_MSInit'
    parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,\
        texture_ref_name=name_texture,loss=['Gram','spectrum'],
        output_img_name=output_img_name,MS_Strat='Init') 
    args = parser.parse_args()
    rs.run_synthesis(args)
    print('End of the autocorr texture test')

# if __name__ == '__main__':
#     # testTexture()
#     testTextureAucorr()
#     # testTextureMSInit()

if __name__ == '__main__':
    # testTexture('/home/thomas/Downloads/CompImg/seg.png')
    for file in glob.glob(os.path.join(dataset_folder, '*.jpg')):

        print("############################################################################################")
        print("######## ", file, " #########")
        print("############################################################################################")
        
        # Simple Gatys
        runTexture(file)

        # Autocorr (no Gram matrix)
        # testTextureAucorr(file)
