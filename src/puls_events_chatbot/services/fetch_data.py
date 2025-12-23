import requests
import pandas as pd

url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records?select=canonicalurl%20as%20URL%2C%20%20title_fr%20as%20Titre%2C%20%20description_fr%20as%20description%2C%20%20longdescription_fr%20as%20description_longue%20%2C%20image%20%2C%20thumbnail%20%2C%20daterange_fr%20as%20date%2C%20firstdate_end%20as%20premier_jour_debut%20%2C%20lastdate_begin%20as%20premier_jour_fin%20%2C%20lastdate_end%20as%20dernier_jour_debut%20%2C%20accessibility_label_fr%20as%20dernier_jour_fin%20%2C%20location_name%20as%20nom_localisation%20%2C%20location_address%20as%20adresse%20%2C%20location_postalcode%20as%20code_postale%20%2C%20location_city%20as%20ville%20%2C%20location_phone%20as%20telephone%20%2C%20location_website%20as%20site_web%20%2C%20location_description_fr%20as%20description_localisation%20%2C%20onlineaccesslink%20as%20lien_acces_en_ligne%20%2C%20age_min%20as%20age_minimum%20%2C%20age_max%20as%20age_maximum%20%2C%20originagenda_title%20as%20source&limit=40&refine=location_region%3A%22Martinique%22&refine=firstdate_begin%3A%222025%22"

def fetch_evenements_publics():

    try:
        print("Fetch events ...")
        response = requests.get(url)
        print("Events retrieve successfully !")

        data = response.json()
        df = pd.json_normalize(data['results'])
        return df
    
    except Exception:
        print(f"Une erreur est survenue durant la connexion a l'api Open Agenda")
        return None
    
def clean_data(df):

    date_columns = ["First date - Beginning","Last date - Beginning", "First date - End", "Last date - End"]
    
    for column in date_columns :
        df[column] = pd.to_datetime(
            df[column], 
            format='ISO8601', 
            errors='coerce' 
        )
    df_unique = df.drop_duplicates(subset=['Description'])
    return df_unique