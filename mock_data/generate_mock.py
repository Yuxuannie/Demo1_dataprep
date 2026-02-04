"""
Mock Data Generator for Demo 1
Generates CSV matching actual MLQC timing data structure
"""

import pandas as pd
import numpy as np
import random


class MockDataGenerator:
    """
    Generates realistic MLQC timing data matching actual CSV structure.
    
    Based on actual columns:
    - arc_pt identifiers (cell names)
    - nominal_delay, lib_sigma_delay_late
    - nominal_tran, lib_sigma_tran_late
    - sigma_by_nominal, early_sigma_by_late_sigma
    - stdev_by_late_sigma, cross_sigma_sec
    """
    
    def __init__(self, n_rows=21818, seed=42):
        self.n_rows = n_rows
        np.random.seed(seed)
        random.seed(seed)
        
        # Cell type library (from real data)
        self.cell_types = [
            ('MUXA2MZ', [1, 2, 4, 8]),      # Mux, various drive strengths
            ('NAND2', [1, 2, 4, 6, 8]),     # NAND gates
            ('NOR2', [1, 2, 4, 6]),         # NOR gates
            ('INV', [1, 2, 4, 6, 8, 12]),   # Inverters
            ('XOR2', [1, 2, 4]),            # XOR gates
            ('AOI21', [1, 2, 4]),           # AND-OR-INVERT
            ('OAI21', [1, 2, 4]),           # OR-AND-INVERT
        ]
        
        # Pin combinations
        self.pins = ['A', 'B', 'C', 'D', 'Z']
    
    def generate_arc_pt_name(self):
        """Generate realistic arc_pt identifier like MUXA2MZD2ZBWP143N117H3P4BCPD"""
        cell_type, drives = random.choice(self.cell_types)
        drive = random.choice(drives)
        
        # Random tech node identifier
        tech = f"BWP143N117H3P4BCPD"
        
        # Random pin combo
        from_pin = random.choice(self.pins[:-1])  # Not Z
        to_pin = 'Z'
        
        return f"{cell_type}D{drive}Z{tech}_{from_pin}#{to_pin}"
    
    def generate(self):
        """Generate complete dataset."""
        
        data = []
        
        for i in range(self.n_rows):
            # Generate arc_pt name
            arc_pt = self.generate_arc_pt_name()
            
            # Extract cell type for realistic timing
            cell_type = arc_pt.split('D')[0]
            
            # Base timing characteristics vary by cell type
            if 'MUX' in cell_type:
                delay_base = np.random.uniform(1000, 2500)
                sigma_factor = np.random.uniform(0.08, 0.15)
                tran_base = np.random.uniform(200, 500)
            elif 'INV' in cell_type:
                delay_base = np.random.uniform(200, 800)
                sigma_factor = np.random.uniform(0.05, 0.10)
                tran_base = np.random.uniform(100, 300)
            elif 'NAND' in cell_type or 'NOR' in cell_type:
                delay_base = np.random.uniform(400, 1200)
                sigma_factor = np.random.uniform(0.06, 0.12)
                tran_base = np.random.uniform(150, 400)
            elif 'XOR' in cell_type:
                delay_base = np.random.uniform(1500, 3000)
                sigma_factor = np.random.uniform(0.10, 0.18)
                tran_base = np.random.uniform(300, 600)
            else:  # AOI, OAI
                delay_base = np.random.uniform(600, 1600)
                sigma_factor = np.random.uniform(0.07, 0.13)
                tran_base = np.random.uniform(180, 450)
            
            # Add some noise
            nominal_delay = delay_base + np.random.normal(0, delay_base * 0.1)
            lib_sigma_delay_late = nominal_delay * sigma_factor
            
            nominal_tran = tran_base + np.random.normal(0, tran_base * 0.1)
            lib_sigma_tran_late = nominal_tran * sigma_factor * 0.8
            
            # Derived metrics
            sigma_by_nominal = lib_sigma_delay_late / nominal_delay
            early_sigma_by_late_sigma = np.random.uniform(0.7, 1.0)
            stdev_by_late_sigma = np.random.uniform(0.85, 1.15)
            cross_sigma_sec = np.random.uniform(-0.1, 0.1)
            
            # Additional delay/tran variants
            from_nominal_by_early_delay = np.random.uniform(0.95, 1.05)
            delay_late_by_delay = np.random.uniform(1.0, 1.3)
            tran_late_by_tran = np.random.uniform(1.0, 1.2)
            
            data.append({
                'arc_pt': arc_pt,
                'nominal_delay': nominal_delay,
                'lib_sigma_delay_late': lib_sigma_delay_late,
                'nominal_tran': nominal_tran,
                'lib_sigma_tran_late': lib_sigma_tran_late,
                'sigma_by_nominal': sigma_by_nominal,
                'early_sigma_by_late_sigma': early_sigma_by_late_sigma,
                'stdev_by_late_sigma': stdev_by_late_sigma,
                'cross_sigma_sec': cross_sigma_sec,
                'from_nominal_by_early_delay': from_nominal_by_early_delay,
                'delay_late_by_delay': delay_late_by_delay,
                'tran_late_by_tran': tran_late_by_tran,
            })
        
        df = pd.DataFrame(data)
        return df
    
    def save(self, filepath):
        """Generate and save to CSV."""
        df = self.generate()
        df.to_csv(filepath, index=False)
        print(f"Generated {len(df)} rows → {filepath}")
        return df


if __name__ == "__main__":
    # Generate test data
    generator = MockDataGenerator(n_rows=21818)
    df = generator.save('/home/claude/demo1_dataprep/mock_data/test_data.csv')
    
    print("\nSample data:")
    print(df.head())
    
    print("\nData statistics:")
    print(df.describe())
    
    print("\nCell type distribution:")
    cell_types = df['arc_pt'].str.extract(r'^([A-Z0-9]+)D')[0]
    print(cell_types.value_counts())
