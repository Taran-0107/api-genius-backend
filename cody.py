
import requests

API_KEY = '+aHWpLr3a7+VdNulqFy7gg==vOcz1uFqtaiSWvlZ'  # Replace with your actual API key
API_BASE_URL = 'https://www.themealdb.com/api/json/v1/1/'

def get_most_popular_foods():
    try:
        # List all categories, areas, and ingredients
        response = requests.get(API_BASE_URL + 'list.php', params={'c':'list', 'a':'list', 'i':'list'})
        
        if response.status_code == 200:
            data = response.json()
            categories = data['categories']
            
            for category in categories:
                print(f"Category: {category['strCategory']}")
                
                # Filter by category
                category_url = API_BASE_URL + 'filter.php'
                params = {'c': category['strCategory']}
                category_response = requests.get(category_url, params=params)
                
                if category_response.status_code == 200:
                    category_data = category_response.json()
                    meals = category_data['meals']
                    
                    for meal in meals:
                        print(f" - {meal['strMeal']}")
                        # Here you can add more details if needed, like meal ID or name
                        
                else:
                    print("Error fetching data for this category")
                    
        else:
            print("Error fetching initial data. Status code:", response.status_code)
            
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    get_most_popular_foods()