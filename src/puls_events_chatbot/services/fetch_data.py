import requests
import pandas as pd

URL_BASE = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
SELECT = "?select=canonicalurl%20as%20URL%2C%20%20title_fr%20as%20Titre%2C%20%20description_fr%20as%20description%2C%20%20longdescription_fr%20as%20description_longue%20%2C%20image%20%2C%20thumbnail%20%2C%20daterange_fr%20as%20date%2C%20firstdate_end%20as%20premier_jour_debut%20%2C%20lastdate_begin%20as%20premier_jour_fin%20%2C%20lastdate_end%20as%20dernier_jour_debut%20%2C%20accessibility_label_fr%20as%20dernier_jour_fin%20%2C%20location_name%20as%20nom_localisation%20%2C%20location_address%20as%20adresse%20%2C%20location_postalcode%20as%20code_postale%20%2C%20location_city%20as%20ville%20%2C%20location_phone%20as%20telephone%20%2C%20location_website%20as%20site_web%20%2C%20location_description_fr%20as%20description_localisation%20%2C%20onlineaccesslink%20as%20lien_acces_en_ligne%20%2C%20age_min%20as%20age_minimum%20%2C%20age_max%20as%20age_maximum%20%2C%20originagenda_title%20as%20source"
FILTERS = "&limit=40&refine=location_countrycode%3A%22FR%22&refine=location_department%3A%22Paris%22&refine=location_city%3A%22Paris%22&refine=firstdate_begin%3A%222026%22&refine=firstdate_end%3A%222026%2F04%22"
url = URL_BASE + SELECT + FILTERS

def fetch_evenements_publics():
    try:
        print("Fetch events ...")
        response = requests.get(url, timeout=15)
        response.raise_for_status() 
        print("Events retrieve successfully !")

        data = response.json()
        df = pd.json_normalize(data['results'])
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Une erreur réseau est survenue avec l'API Open Agenda : {e}")
        return None
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors de la récupération : {e}")
        return None
    
def clean_data(df):
    if df is None or df.empty:
        return df

    if 'description' in df.columns:
        df = df.dropna(subset=['description'])
        df = df[df['description'].astype(str).str.strip() != '']
        df = df.drop_duplicates(subset=['description'])
        
    date_columns = ["premier_jour_debut", "premier_jour_fin", "dernier_jour_debut", "dernier_jour_fin"]
    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column], 
                format='ISO8601', 
                errors='coerce' 
            )

    return df