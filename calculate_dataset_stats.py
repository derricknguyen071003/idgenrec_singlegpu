#!/usr/bin/env python3
"""
Calculate dataset statistics: #Users, #Items, #Interactions, #Relations,
Interaction Density, and Relation Density for specified datasets.
"""

import os
from collections import defaultdict

def calculate_dataset_stats(data_path, dataset_name):
    """
    Calculate statistics for a dataset.
    
    Returns:
        dict with keys: num_users, num_items, num_interactions, num_relations,
        interaction_density, relation_density
    """
    dataset_path = os.path.join(data_path, dataset_name)
    
    # Load user-item interactions
    user_sequence_file = os.path.join(dataset_path, 'user_sequence.txt')
    if not os.path.exists(user_sequence_file):
        raise FileNotFoundError(f"user_sequence.txt not found in {dataset_path}")
    
    users = set()
    items = set()
    num_interactions = 0
    
    with open(user_sequence_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            user_id = parts[0]
            item_ids = parts[1:]
            
            users.add(user_id)
            items.update(item_ids)
            num_interactions += len(item_ids)
    
    num_users = len(users)
    num_items = len(items)
    
    # Load friend relations
    friend_sequence_file = os.path.join(dataset_path, 'friend_sequence.txt')
    num_relations_raw = 0
    relation_users = set()
    edges = set()  # To count unique edges for undirected graph
    
    if os.path.exists(friend_sequence_file):
        with open(friend_sequence_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                user_id = parts[0]
                friend_ids = parts[1:]
                
                relation_users.add(user_id)
                relation_users.update(friend_ids)
                num_relations_raw += len(friend_ids)
                
                # Store edges (both directions for undirected graph)
                for friend_id in friend_ids:
                    # Store edge in canonical form (smaller ID first) to avoid duplicates
                    edge = tuple(sorted([user_id, friend_id]))
                    edges.add(edge)
        
        # For undirected graph, each edge appears twice in the file
        # So unique edges = num_relations_raw / 2, or we can use the set size
        num_relations = len(edges)
    else:
        print(f"Warning: friend_sequence.txt not found in {dataset_path}, setting relations to 0")
        num_relations = 0
    
    # Calculate densities
    # Interaction Density = #Interactions / (#Users * #Items)
    if num_users > 0 and num_items > 0:
        interaction_density = num_interactions / (num_users * num_items)
    else:
        interaction_density = 0.0
    
    # Relation Density = #Unique Relations / (#Users * (#Users - 1) / 2) for undirected graph
    # For undirected graph, max possible edges = n*(n-1)/2
    if num_users > 1:
        max_possible_relations = num_users * (num_users - 1) / 2
        relation_density = num_relations / max_possible_relations if max_possible_relations > 0 else 0.0
    else:
        relation_density = 0.0
    
    return {
        'num_users': num_users,
        'num_items': num_items,
        'num_interactions': num_interactions,
        'num_relations': num_relations,
        'interaction_density': interaction_density,
        'relation_density': relation_density
    }

def main():
    data_path = 'rec_datasets'
    datasets = ['delicious', 'yelp300_5_5', 'lastfm']
    
    print("Dataset Statistics")
    print("=" * 100)
    print(f"{'Dataset':<20} {'#Users':<10} {'#Items':<10} {'#Interactions':<15} {'#Relations':<12} {'Interaction Density':<20} {'Relation Density':<20}")
    print("-" * 100)
    
    results = []
    for dataset in datasets:
        try:
            stats = calculate_dataset_stats(data_path, dataset)
            results.append((dataset, stats))
            
            # Format densities as percentages with 2 decimal places
            interaction_density_pct = stats['interaction_density'] * 100
            relation_density_pct = stats['relation_density'] * 100
            
            print(f"{dataset:<20} {stats['num_users']:<10} {stats['num_items']:<10} {stats['num_interactions']:<15} "
                  f"{stats['num_relations']:<12} {interaction_density_pct:>6.2f}%{'':<13} {relation_density_pct:>6.2f}%{'':<13}")
        except Exception as e:
            print(f"{dataset:<20} ERROR: {str(e)}")
    
    print("=" * 100)
    
    # Also save to a file
    output_file = os.path.join(data_path, 'dataset_statistics.txt')
    with open(output_file, 'w') as f:
        f.write("Dataset Statistics\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Dataset':<20} {'#Users':<10} {'#Items':<10} {'#Interactions':<15} {'#Relations':<12} "
                f"{'Interaction Density':<20} {'Relation Density':<20}\n")
        f.write("-" * 100 + "\n")
        
        for dataset, stats in results:
            # Format densities as percentages with 2 decimal places
            interaction_density_pct = stats['interaction_density'] * 100
            relation_density_pct = stats['relation_density'] * 100
            
            f.write(f"{dataset:<20} {stats['num_users']:<10} {stats['num_items']:<10} {stats['num_interactions']:<15} "
                   f"{stats['num_relations']:<12} {interaction_density_pct:>6.2f}%{'':<13} {relation_density_pct:>6.2f}%{'':<13}\n")
        
        f.write("=" * 100 + "\n")
    
    print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main()

