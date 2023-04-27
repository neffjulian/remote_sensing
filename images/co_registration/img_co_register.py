from pathlib import Path

from arosics import COREG

# def co_register_images(year: str, month: str, index: int):
def co_register_images():
    planet_image = Path('images/planet_scope/22_april/0000/20220428_101547_22_2426\plot.png')
    sentinel_image = Path('images/sentinel_2/22_apr/0000/plot.png')

    path_out = Path('images/co_registration')
    CR = COREG(planet_image, sentinel_image, path_out=path_out)
    CR.calculate_spatial_shifts()
