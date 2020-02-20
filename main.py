from process import extract_glasses_location, train_or_load_character_recognition_model

'''
1. Treba napraviti model i sacuvati u folderu serialized model
2. Sacuvani model ne dirati nakon podesavanja, vec ga samo proslediti kao prvi parametar funkcije
   extract_glasses_location
3. Funkcija bi trebalo da vrati lokaciju casa po trenutnoj zamisli
'''
extract_glasses_location(None, 'case/1.jpg')
