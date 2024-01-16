# -*- coding: utf-8 -*-

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from tqdm import tqdm
import cv2
import tensorflow as tf
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import ImageFont, ImageDraw
import warnings

def update_sheet(worksheet, lien, text_content):
    # Trouver la ligne correspondant au lien dans la feuille de calcul
    cell_list = worksheet.findall(lien)
    if not cell_list:
        print(f"Le lien {lien} n'a pas été trouvé dans la feuille de calcul.")
        return

    cell = cell_list[0]

    # Mettre à jour la colonne AL avec le texte extrait, en divisant le texte tous les 20000 caractères
    col_al_text = [text_content[i:i + 20000] for i in range(0, len(text_content), 20000)]

    for i, text_part in enumerate(col_al_text):
        col_al_cell = worksheet.cell(cell.row, 38 + i)  # 38 correspond à la colonne AL (1-indexed)
        col_al_cell.value = text_part

        # Mettre à jour la feuille de calcul
        worksheet.update_cells([col_al_cell])

    print(f"Texte extrait mis à jour dans la feuille de calcul pour le lien {lien}.")

def apply_ocr(pdf_path):
    # Charger le fichier PDF
    doc = DocumentFile.from_pdf(pdf_path)

    # Initialiser le modèle OCR
    det_arch = "db_resnet50" if doc[0].shape[0] > 1000 else "linknet_resnet18_rotation"
    reco_arch = "crnn_mobilenet_v3_small"
    predictor = ocr_predictor(det_arch, reco_arch, pretrained=True, assume_straight_pages=(det_arch != "linknet_resnet18_rotation"))

    # Charger la police avec une taille spécifiée
    font_size = 12
    font_family = "arial"  # Vous pouvez choisir une autre famille de polices si nécessaire
    font = ImageFont.truetype(font_family, font_size)

    # Accumuler le texte de toutes les pages
    total_text_content = ""

    # Parcourir les pages du document
    for page_idx in tqdm(range(len(doc)), desc=f"Applying OCR to {pdf_path}"):
        # Transmettre l'image au modèle
        processed_batches = predictor.det_predictor.pre_processor([doc[page_idx]])
        out = predictor.det_predictor.model(processed_batches[0], return_model_output=True)
        seg_map = out["out_map"]
        seg_map = tf.squeeze(seg_map[0, ...], axis=[2])
        seg_map = cv2.resize(seg_map.numpy(), (doc[page_idx].shape[1], doc[page_idx].shape[0]),
                             interpolation=cv2.INTER_LINEAR)

        # Progress bar pour simuler l'analyse OCR
        progress_bar = tqdm(total=100, position=0, leave=False)
        for i in range(100):
            # Simuler un processus OCR chronophage
            progress_bar.update(1)
        progress_bar.close()

        # Obtenir la sortie OCR
        out = predictor([doc[page_idx]])

        # Reconstituer la page sous la page d'entrée
        if det_arch != "linknet_resnet18_rotation":
            img = out.pages[0].synthesize()

        # Afficher le texte extrait
        for block in out.pages[0].blocks:
            for line in block.lines:
                for word in line.words:
                    total_text_content += word.value + " "

    print(f"**Texte extrait de toutes les pages :** {total_text_content}")
    return total_text_content

# Ignorer les avertissements pendant l'exécution de certaines parties du code
warnings.filterwarnings("ignore")

def main():
    # Autorisation Google Sheets
    scope_sheets = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_sheets = ServiceAccountCredentials.from_json_keyfile_name("C:/Users/Ruth/Downloads/test/automatisation-contrat-pdf-0a30bbe0f037.json", scope_sheets)
    gc_sheets = gspread.authorize(creds_sheets)

    # Ouverture de la feuille de calcul par clé
    sh = gc_sheets.open_by_key("1cxnEdB_M0CwGxiKuLX_SBphx2E9c1rdtNtnl4-cHsCg")
    wks_source = sh.worksheet("Source (ne pas toucher)")

    # Récupère les valeurs de la colonne D à partir de la ligne 5
    contrats = wks_source.col_values(4)[4:]

    # Initialisation de l'authentification Google Drive
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    # Dossier de destination pour les fichiers téléchargés
    dossier = "C:/Users/Ruth/Downloads"

    # Télécharge les fichiers PDF localement
    for lien in contrats:
        # Extrait l'ID du fichier à partir du lien
        file_id = lien.split('/')[5]
        print(f"ID du fichier : {file_id}")

        # Récupère l'objet file correspondant
        file = drive.CreateFile({'id': file_id})
        file.GetContentFile(os.path.join(dossier, file['title']))

        # Vérifie si le type de fichier est un document Google
        if file['mimeType'] == 'application/pdf':
            # Télécharge le fichier à partir de son ID
            file = drive.CreateFile({'id': file_id})
            file.GetContentFile(os.path.join(dossier, file['title']))
            file_name = file['title'].replace(" ", "_").replace("/", "_")  # Modification du nom du fichier

            os.rename(os.path.join(dossier, file['title']), os.path.join(dossier, file_name))
            print(f"Fichier {file_name} téléchargé avec succès.")

            # Appliquer l'OCR au fichier PDF
            text_content = apply_ocr(os.path.join(dossier, file_name))

            # Check if there is text content before updating the sheet
            if text_content:
                # Mettre à jour la feuille de calcul avec le texte extrait
                update_sheet(wks_source, lien, text_content)
            else:
                print(f"Aucun texte extrait pour le lien {lien}.")

if __name__ == "__main__":
    main()
