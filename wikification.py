import sys, http.client, urllib.request, urllib.parse, urllib.error, json, re

from pprint import pprint

### This method looks up string in dbpedia.
def wikiLookup(ne): 

    # Get the data from dbpedia
    query = ne.replace(" ", "_")
    query = query.replace("`", "")
    url_data = None
    try:
        url_data = urllib.request.urlopen("http://lookup.dbpedia.org/api/search/KeywordSearch?QueryString="+query)
        url_data = url_data.read().decode("utf-8")
    except Exception as e:
        print("error")
        print(query)
        print(e)
        return None

    if url_data is None :
        print( "Failed to get data ... Can not proceed." )
        sys.exit()

    # Find all labels
    labels = re.findall("<Label>(.+)</Label>", url_data)

    # If named entity contains digits followed by other characters, make it a location
    if re.match("\d+.+", ne) or re.match(".+\d+", ne):
        return "LOCATION"

    # Determine the type from the labels
    personWords = ["person"]
    locationWords = ["place", "location"]
    orgWikiWords = ["organization", "organisation"]
    for l in labels:
        if l in personWords:
            return "PERSON"
        elif l in orgWikiWords:
            return "ORGANIZATION"
        elif l in locationWords:
            return "LOCATION"


    return None
