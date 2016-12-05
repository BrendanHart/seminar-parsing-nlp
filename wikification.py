import sys, http.client, urllib.request, urllib.parse, urllib.error, json, re

from pprint import pprint

def wikiLookup(ne): 

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

    labels = re.findall("<Label>(.+)</Label>", url_data)

    if re.match("\d+.+", ne) or re.match(".+\d+", ne):
        return "LOCATION"

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
