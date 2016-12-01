from bs4 import BeautifulSoup
import requests
import time

hdr = {'User-Agent': 'scraping match ids, pls don\'t restrict me, -Rizhiy-'}

players = open('player_ids.txt', 'r').readlines()

matches_file = open('matches_ids.txt', 'a')

for player in players:
    print(player)
    page_number = 0
    current_page = "start"
    prev_page = ""
    while current_page != prev_page:
        page_number += 1
        time.sleep(5)
        url = 'http://www.dotabuff.com/esports/players/' + player.rstrip('\n') + '/matches?page=' + str(page_number)
        r = requests.get(url, headers=hdr)

        data = r.text

        soup = BeautifulSoup(data, 'lxml')
        prev_page = current_page
        current_page = soup.select(
            'body > div.container-outer > div.container-inner > div.content-inner > div.row-12 > div.col-8 > section > article > div')[
            0].text
        lost_matches = soup.select('.lost')
        won_matches = soup.select('.won')

        matches = lost_matches + won_matches

        for link in matches:
            print(link['href'].split('/')[-1], file=matches_file)
