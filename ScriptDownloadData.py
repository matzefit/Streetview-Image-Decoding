from cityscapesscripts.download import downloader

#registered at cityscapes-dataset.com

session = downloader.login()
downloader.get_available_packages(session=session)

#data choice
print('downloading gtfine and leftImg8bit packages ... \n')

package_list = {'gtFine_trainvaltest.zip', 'leftImg8bit_trainvaltest.zip'}
downloader.download_packages(session=session, package_names=package_list, destination_path=r'C:\Users\Citylab\Documents\PythonProjects\ComputerVision\SemesterProject\Data')