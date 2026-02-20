#!/usr/bin/env python3
"""
Funda HTML Parser
Extracts apartment listings from downloaded Funda search result HTML files.

Usage:
    python src/parse_funda.py <subfolder_name>
    
Example:
    python src/parse_funda.py nellestein
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime


class NuxtDataParser:
    """Parser for Nuxt's reference-based JSON data format"""
    
    def __init__(self, data: List[Any]):
        self.data = data
    
    def resolve(self, value: Any, depth: int = 0, max_depth: int = 10) -> Any:
        """Recursively resolve references in the Nuxt data structure with depth limit"""
        # Prevent infinite recursion
        if depth > max_depth:
            return value
            
        if isinstance(value, int):
            # Integer is an index reference
            if 0 <= value < len(self.data):
                return self.resolve(self.data[value], depth + 1, max_depth)
            return value
        elif isinstance(value, list):
            # Check if it's a special reference like ['Ref', index] or ['Reactive', index]
            if len(value) == 2 and isinstance(value[0], str) and value[0] in ['Ref', 'Reactive', 'ShallowReactive']:
                return self.resolve(value[1], depth + 1, max_depth)
            # Otherwise resolve each element
            return [self.resolve(item, depth + 1, max_depth) for item in value]
        elif isinstance(value, dict):
            # Resolve all values in the dictionary
            return {k: self.resolve(v, depth + 1, max_depth) for k, v in value.items()}
        else:
            # Primitive value, return as-is
            return value
    
    def get(self, index: int) -> Any:
        """Get value at index without full resolution"""
        if 0 <= index < len(self.data):
            return self.data[index]
        return None


class FundaHTMLParser:
    """Parser for Funda HTML files"""
    
    def __init__(self):
        self.parser = None
    
    def extract_nuxt_data(self, html_content: str) -> Optional[List[Any]]:
        """Extract the __NUXT_DATA__ JSON from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        script_tag = soup.find('script', {'id': '__NUXT_DATA__'})
        
        if not script_tag or not script_tag.string:
            return None
        
        try:
            data = json.loads(script_tag.string)
            return data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None
    
    def find_listings(self, data: List[Any], debug: bool = False) -> Optional[List[int]]:
        """Find and extract listing indices from the Nuxt data structure"""
        if not data or len(data) < 5:
            if debug:
                print(f"  DEBUG: Data too small or empty")
            return None
        
        # Look for a dict that has 'listings' key directly
        # This is typically found within the first 200 items
        listings_ref = None
        for i, item in enumerate(data[:200]):
            if isinstance(item, dict) and 'listings' in item and 'totalListingsCount' in item:
                listings_ref = item['listings']
                if debug:
                    print(f"  DEBUG: Found listings dict at index {i}")
                    print(f"  DEBUG: listings_ref = {listings_ref}")
                break
        
        if listings_ref is None:
            if debug:
                print(f"  DEBUG: No listings dict found")
            return None
        
        # listings_ref is typically ['Ref', index] or ['Reactive', index] or just an int
        if isinstance(listings_ref, list) and len(listings_ref) == 2:
            listings_idx = listings_ref[1]
            if debug:
                print(f"  DEBUG: listings_ref is list, using index {listings_idx}")
        elif isinstance(listings_ref, int):
            listings_idx = listings_ref
            if debug:
                print(f"  DEBUG: listings_ref is int: {listings_idx}")
        else:
            if debug:
                print(f"  DEBUG: listings_ref is unexpected type: {type(listings_ref)}")
            return None
        
        # Get the listings array - it's another reference
        if listings_idx >= len(data):
            if debug:
                print(f"  DEBUG: listings_idx {listings_idx} out of range")
            return None
            
        listings_container = data[listings_idx]
        if debug:
            print(f"  DEBUG: listings_container type: {type(listings_container)}")
            if isinstance(listings_container, list):
                print(f"  DEBUG: listings_container: {listings_container}")
        
        if isinstance(listings_container, list) and len(listings_container) == 2:
            actual_listings_idx = listings_container[1]
            if debug:
                print(f"  DEBUG: Using actual_listings_idx: {actual_listings_idx}")
        elif isinstance(listings_container, int):
            actual_listings_idx = listings_container
            if debug:
                print(f"  DEBUG: listings_container is int: {actual_listings_idx}")
        else:
            if debug:
                print(f"  DEBUG: listings_container unexpected format")
            return None
        
        # Now get the actual array of listing indices
        if actual_listings_idx >= len(data):
            if debug:
                print(f"  DEBUG: actual_listings_idx {actual_listings_idx} out of range")
            return None
            
        listings_array = data[actual_listings_idx]
        
        if debug:
            print(f"  DEBUG: listings_array type: {type(listings_array)}")
            if isinstance(listings_array, list):
                print(f"  DEBUG: listings_array length: {len(listings_array)}")
                print(f"  DEBUG: First few items: {listings_array[:5]}")
        
        # Check if it's another reference wrapper like ['Reactive', index]
        if isinstance(listings_array, list) and len(listings_array) == 2 and isinstance(listings_array[0], str):
            final_idx = listings_array[1]
            if debug:
                print(f"  DEBUG: listings_array is a wrapper, using final index: {final_idx}")
            if final_idx >= len(data):
                if debug:
                    print(f"  DEBUG: final_idx {final_idx} out of range")
                return None
            listings_array = data[final_idx]
            if debug:
                print(f"  DEBUG: Final listings_array type: {type(listings_array)}")
                if isinstance(listings_array, list):
                    print(f"  DEBUG: Final listings_array length: {len(listings_array)}")
                    print(f"  DEBUG: First few items: {listings_array[:5]}")
        
        if not isinstance(listings_array, list):
            if debug:
                print(f"  DEBUG: listings_array is not a list")
            return None
        
        # Return the list of listing indices
        return listings_array
    
    def extract_listing_info(self, listing: Dict, parser: NuxtDataParser) -> Dict[str, Any]:
        """Extract relevant information from a single listing"""
        result = {}
        
        # ID
        result['id'] = listing.get('id')
        
        # Address information
        address = parser.resolve(listing.get('address', {}))
        if isinstance(address, dict):
            result['street_name'] = address.get('street_name', '')
            result['house_number'] = address.get('house_number', '')
            result['postal_code'] = address.get('postal_code', '')
            result['city'] = address.get('city', '')
            result['neighbourhood'] = address.get('neighbourhood', '')
            
            # Construct full address
            street = result['street_name']
            number = result['house_number']
            result['address'] = f"{street} {number}".strip() if street and number else ''
        else:
            result['street_name'] = ''
            result['house_number'] = ''
            result['postal_code'] = ''
            result['city'] = ''
            result['neighbourhood'] = ''
            result['address'] = ''
        
        # Price information
        price_data = parser.resolve(listing.get('price', {}), max_depth=5)
        if isinstance(price_data, dict):
            selling_price = price_data.get('selling_price')
            # selling_price might be a list with one element
            if isinstance(selling_price, list) and len(selling_price) > 0:
                result['price'] = selling_price[0]
            else:
                result['price'] = selling_price
        else:
            result['price'] = None
        
        # Floor area (square meters) - custom resolution to avoid over-resolving
        # Pattern: index -> [index] -> actual_value
        floor_area = listing.get('floor_area')
        if isinstance(floor_area, int) and 0 <= floor_area < len(parser.data):
            floor_area = parser.data[floor_area]
            # If it's a list with one element, get that element
            if isinstance(floor_area, list) and len(floor_area) > 0:
                floor_area_idx = floor_area[0]
                # Resolve one more level if it's an index
                if isinstance(floor_area_idx, int) and 0 <= floor_area_idx < len(parser.data):
                    floor_area = parser.data[floor_area_idx]
                else:
                    floor_area = floor_area_idx
        result['floor_area'] = floor_area
        
        # Number of rooms and bedrooms - dereference if they're indices (single level only)
        number_of_rooms = listing.get('number_of_rooms')
        if isinstance(number_of_rooms, int) and 0 <= number_of_rooms < len(parser.data):
            number_of_rooms = parser.data[number_of_rooms]
        result['number_of_rooms'] = number_of_rooms
        
        number_of_bedrooms = listing.get('number_of_bedrooms')
        if isinstance(number_of_bedrooms, int) and 0 <= number_of_bedrooms < len(parser.data):
            number_of_bedrooms = parser.data[number_of_bedrooms]
        result['number_of_bedrooms'] = number_of_bedrooms
        
        # Energy label - resolve if it's an index
        energy_label = listing.get('energy_label', '')
        if isinstance(energy_label, int):
            energy_label = parser.resolve(energy_label, max_depth=2)
        result['energy_label'] = energy_label if energy_label else ''
        
        # Status (available, negotiations, sold, etc.) - resolve if it's an index
        status = listing.get('status', '')
        if isinstance(status, int):
            status = parser.resolve(status, max_depth=2)
        result['status'] = status if status else ''
        
        # Object type - dereference if it's an index (single level only)
        object_type = listing.get('object_type', '')
        if isinstance(object_type, int) and 0 <= object_type < len(parser.data):
            object_type = parser.data[object_type]
        result['object_type'] = object_type if object_type else ''
        
        # URL - dereference if it's an index (single level only)
        url = listing.get('object_detail_page_relative_url', '')
        if isinstance(url, int) and 0 <= url < len(parser.data):
            url = parser.data[url]
        result['url'] = f"https://www.funda.nl{url}" if url else ''
        
        # Publish date - dereference if it's an index (single level only)
        publish_date = listing.get('publish_date', '')
        if isinstance(publish_date, int) and 0 <= publish_date < len(parser.data):
            publish_date = parser.data[publish_date]
        result['publish_date'] = publish_date if publish_date else ''
        
        return result
    
    def parse_file(self, file_path: Path, debug: bool = False) -> List[Dict[str, Any]]:
        """Parse a single HTML file and extract all listings"""
        print(f"  Processing: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except Exception as e:
            print(f"  Error reading file: {e}")
            return []
        
        # Extract Nuxt data
        data = self.extract_nuxt_data(html_content)
        if not data:
            print(f"  Warning: No Nuxt data found in {file_path.name}")
            return []
        
        if debug:
            print(f"  DEBUG: Extracted data array with {len(data)} items")
        
        # Find listing indices
        listing_indices = self.find_listings(data, debug=debug)
        if not listing_indices:
            print(f"  Warning: No listings found in {file_path.name}")
            return []
        
        if debug:
            print(f"  DEBUG: Found {len(listing_indices)} listing indices")
        
        # Parse each listing
        parser = NuxtDataParser(data)
        results = []
        for listing_idx in listing_indices:
            if isinstance(listing_idx, int) and 0 <= listing_idx < len(data):
                listing = data[listing_idx]
                if isinstance(listing, dict):
                    listing_info = self.extract_listing_info(listing, parser)
                    results.append(listing_info)
                elif debug:
                    print(f"  DEBUG: Listing at index {listing_idx} is not a dict: {type(listing)}")
            elif debug:
                print(f"  DEBUG: Invalid listing index: {listing_idx}")
        
        print(f"  Found {len(results)} listings")
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Parse Funda HTML files and extract apartment listings'
    )
    parser.add_argument(
        'subfolder',
        help='Subfolder name in data/funda/ to process (e.g., "nellestein")'
    )
    parser.add_argument(
        '--output-format',
        choices=['csv', 'json', 'both'],
        default='csv',
        help='Output format (default: csv)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    input_dir = base_dir / 'data' / 'funda' / args.subfolder
    output_dir = base_dir / 'outputs' / 'funda'
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all HTML files
    html_files = list(input_dir.glob('*.html'))
    if not html_files:
        print(f"Error: No HTML files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(html_files)} HTML file(s) in {input_dir}")
    print()
    
    # Parse all files
    funda_parser = FundaHTMLParser()
    all_listings = []
    
    for html_file in sorted(html_files):
        listings = funda_parser.parse_file(html_file, debug=args.debug)
        all_listings.extend(listings)
    
    print()
    print(f"Total listings extracted: {len(all_listings)}")
    
    if not all_listings:
        print("No listings found. Exiting.")
        sys.exit(0)
    
    # Remove duplicates based on ID
    unique_listings = {}
    for listing in all_listings:
        listing_id = listing.get('id')
        if listing_id and listing_id not in unique_listings:
            unique_listings[listing_id] = listing
    
    print(f"Unique listings: {len(unique_listings)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(list(unique_listings.values()))
    
    # Sort by price
    if 'price' in df.columns:
        df = df.sort_values('price')
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = output_dir / f"funda_{args.subfolder}_{timestamp}"
    
    # Save outputs
    if args.output_format in ['csv', 'both']:
        csv_file = output_base.with_suffix('.csv')
        df.to_csv(csv_file, index=False)
        print(f"\nSaved CSV to: {csv_file}")
    
    if args.output_format in ['json', 'both']:
        json_file = output_base.with_suffix('.json')
        df.to_json(json_file, orient='records', indent=2)
        print(f"Saved JSON to: {json_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total listings: {len(df)}")
    
    if 'price' in df.columns and not df['price'].isna().all():
        print(f"Price range: €{df['price'].min():,.0f} - €{df['price'].max():,.0f}")
        print(f"Average price: €{df['price'].mean():,.0f}")
        print(f"Median price: €{df['price'].median():,.0f}")
    
    if 'floor_area' in df.columns and not df['floor_area'].isna().all():
        print(f"Floor area range: {df['floor_area'].min():.0f} - {df['floor_area'].max():.0f} m²")
        print(f"Average floor area: {df['floor_area'].mean():.1f} m²")
    
    if 'status' in df.columns:
        print(f"\nStatus distribution:")
        for status, count in df['status'].value_counts().items():
            print(f"  {status}: {count}")
    
    if 'energy_label' in df.columns:
        print(f"\nEnergy label distribution:")
        for label, count in df['energy_label'].value_counts().head(10).items():
            print(f"  {label}: {count}")
    
    print("="*60)


if __name__ == '__main__':
    main()
