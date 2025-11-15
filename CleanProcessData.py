import pandas as pd
import numpy as np
import os
from datetime import datetime

class FourthDownDataProcessor:
    def __init__(self):
        self.processed_data = None
        
    def load_multiple_seasons(self, base_path=".", years=range(2019, 2025)):
        """
        Load multiple seasons of NFL play-by-play data
        """
        all_seasons = []
        
        for year in years:
            filename = f"./data/pbp-{year}.csv"
            filepath = os.path.join(base_path, filename)
            
            try:
                print(f"Loading {filename}...")
                df_season = pd.read_csv(filepath, low_memory=False)
                df_season['SeasonYear'] = year
                all_seasons.append(df_season)
                print(f"✓ Successfully loaded {filename} with {len(df_season)} rows")
                
            except FileNotFoundError:
                print(f"✗ File not found: {filename}")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
        
        if not all_seasons:
            raise ValueError("No data files were successfully loaded")
        
        # Combine all seasons
        combined_data = pd.concat(all_seasons, ignore_index=True, sort=False)
        print(f"\n📊 Combined dataset: {len(combined_data)} rows from {years[0]}-{years[-1]}")
        
        return combined_data

    def clean_and_process_data(self, df):
        """
        Main data cleaning and processing pipeline for 4th down analysis
        """
        print("\n🧹 Starting data cleaning and processing...")
        original_size = len(df)
        
        # Create a clean copy
        df_clean = df.copy()
        
        # 1. Filter for 4th down plays only
        fourth_down_mask = df_clean['Down'] == 4
        df_clean = df_clean[fourth_down_mask].copy()
        print(f"✓ Filtered to 4th down plays: {len(df_clean)} rows")
        
        # 2. Handle missing values in key columns
        key_numeric_columns = ['Down', 'ToGo', 'YardLine', 'YardLineFixed', 'Yards', 'Quarter', 'Minute', 'Second']
        for col in key_numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                if col == 'Down':  # For Down, use mode since it should be 4
                    df_clean[col].fillna(4, inplace=True)
                else:  # For others, use median
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # 3. Create contextual features
        df_clean = self.create_contextual_features(df_clean)
        
        # 4. Process play types and outcomes
        df_clean = self.process_play_types(df_clean)
        
        # 5. Create success indicators
        df_clean = self.create_success_indicators(df_clean)
        
        # 6. Select final columns for analysis
        df_clean = self.select_final_columns(df_clean)
        
        # 7. Final data quality check
        df_clean = self.final_quality_check(df_clean)
        
        print(f"🎯 Final processed dataset: {len(df_clean)} fourth down plays")
        print(f"📈 Data reduction: {original_size} → {len(df_clean)} rows ({(1 - len(df_clean)/original_size)*100:.1f}% reduction)")
        
        return df_clean

    def create_contextual_features(self, df):
        """
        Create all contextual features for 4th down analysis
        """
        df_clean = df.copy()
        
        # 1. Field Position Features
        if 'YardLineFixed' in df_clean.columns:
            df_clean['FieldPosition'] = df_clean['YardLineFixed'] / 100
            df_clean['RedZone'] = (df_clean['YardLineFixed'] >= 80).astype(int)
            df_clean['OpponentTerritory'] = (df_clean['YardLineFixed'] > 50).astype(int)
            df_clean['OwnTerritory'] = (df_clean['YardLineFixed'] <= 50).astype(int)
            print("✓ Created field position features")
        
        # 2. Down & Distance Features
        if 'ToGo' in df_clean.columns:
            df_clean['YardsToGo'] = df_clean['ToGo']
            df_clean['ShortYardage'] = (df_clean['ToGo'] <= 2).astype(int)  # 4th & 1-2
            df_clean['MediumYardage'] = ((df_clean['ToGo'] >= 3) & (df_clean['ToGo'] <= 6)).astype(int)  # 4th & 3-6
            df_clean['LongYardage'] = (df_clean['ToGo'] >= 7).astype(int)  # 4th & 7+
            df_clean['DownToGoRatio'] = df_clean['ToGo'] / (df_clean['Down'] + 1)
            print("✓ Created down & distance features")
        
        # 3. Game Context Features
        if 'Quarter' in df_clean.columns:
            df_clean['LateGame'] = (df_clean['Quarter'] >= 4).astype(int)
            df_clean['Overtime'] = (df_clean['Quarter'] > 4).astype(int)
        
        # Game time calculation (minutes elapsed)
        if 'Quarter' in df_clean.columns and 'Minute' in df_clean.columns:
            df_clean['GameTime'] = (df_clean['Quarter'] - 1) * 15 + (15 - df_clean['Minute'])
            print("✓ Created game time features")
        
        
        print("Score differential features set as placeholders - enhance with game context data")
        
        return df_clean

    def process_play_types(self, df):
        """
        Process and standardize play type indicators
        """
        df_clean = df.copy()
        
        # Ensure binary indicators are properly formatted
        binary_columns = ['IsRush', 'IsPass', 'IsIncomplete', 'IsTouchdown', 'IsSack', 
                         'IsInterception', 'IsFumble', 'IsPenalty', 'IsTwoPointConversion']
        
        for col in binary_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
        
        # Create decision type based on play characteristics
        conditions = [
            df_clean['IsRush'] == 1,
            df_clean['IsPass'] == 1,
            df_clean['PlayType'] == 'FIELD_GOAL',  # Check PlayType column
            df_clean['PlayType'] == 'PUNT',        # Check PlayType column
            (df_clean['Description'].str.contains('field goal', case=False, na=False)) |
            (df_clean['Description'].str.contains('kick', case=False, na=False)),
            (df_clean['Description'].str.contains('punt', case=False, na=False))
        ]
        
        choices = ['Rush', 'Pass', 'FieldGoal', 'Punt', 'FieldGoal', 'Punt']
        df_clean['DecisionType'] = np.select(conditions, choices, default='Unknown')
        
        # Create binary indicators for each decision type
        df_clean['IsGoForIt'] = ((df_clean['DecisionType'] == 'Rush') | 
                                (df_clean['DecisionType'] == 'Pass')).astype(int)
        df_clean['IsFieldGoalAttempt'] = (df_clean['DecisionType'] == 'FieldGoal').astype(int)
        df_clean['IsPuntAttempt'] = (df_clean['DecisionType'] == 'Punt').astype(int)
        
        print("✓ Processed play type and decision features")
        
        return df_clean

    def create_success_indicators(self, df):
        """
        Create success/failure indicators for 4th down attempts
        """
        df_clean = df.copy()
        
        # 1. Conversion success (for go-for-it attempts)
        if 'Yards' in df_clean.columns and 'ToGo' in df_clean.columns:
            df_clean['ConversionSuccess'] = ((df_clean['Yards'] >= df_clean['ToGo']) & 
                                           (df_clean['IsGoForIt'] == 1)).astype(int)
            # Set non-go-for-it attempts to NaN (they didn't attempt conversion)
            df_clean.loc[df_clean['IsGoForIt'] == 0, 'ConversionSuccess'] = np.nan
        
        # 2. Field goal success
        if 'IsTouchdown' in df_clean.columns:  # Using IsTouchdown as proxy for field goal success
            df_clean['FieldGoalSuccess'] = ((df_clean['IsTouchdown'] == 1) & 
                                          (df_clean['IsFieldGoalAttempt'] == 1)).astype(int)
            df_clean.loc[df_clean['IsFieldGoalAttempt'] == 0, 'FieldGoalSuccess'] = np.nan
        
        # 3. Overall play success (no turnover, positive outcome)
        turnover_conditions = (df_clean['IsInterception'] == 1) | (df_clean['IsFumble'] == 1)
        penalty_conditions = (df_clean['IsPenalty'] == 1)
        
        df_clean['PlaySuccess'] = (~turnover_conditions & ~penalty_conditions).astype(int)
        
        print("✓ Created success indicators")
        
        return df_clean

    def select_final_columns(self, df):
        """
        Select and organize final columns for the output CSV
        """
        # Define column groups for organized output
        identifier_columns = [
            'GameId', 'GameDate', 'SeasonYear', 'OffenseTeam', 'DefenseTeam', 'Description'
        ]
        
        core_context_columns = [
            'Down', 'YardsToGo', 'YardLineFixed', 'FieldPosition', 'RedZone', 
            'OpponentTerritory', 'OwnTerritory', 'Quarter', 'GameTime', 'LateGame', 'Overtime'
        ]
        
        yardage_situation_columns = [
            'ShortYardage', 'MediumYardage', 'LongYardage', 'DownToGoRatio'
        ]
        
        decision_type_columns = [
            'DecisionType', 'IsGoForIt', 'IsFieldGoalAttempt', 'IsPuntAttempt',
            'IsRush', 'IsPass', 'PlayType', 'Formation'
        ]
        
        outcome_columns = [
            'Yards', 'ConversionSuccess', 'FieldGoalSuccess', 'PlaySuccess',
            'IsTouchdown', 'IsIncomplete', 'IsSack', 'IsInterception', 'IsFumble'
        ]
        
        # Combine all desired columns
        all_desired_columns = (identifier_columns + core_context_columns + 
                              yardage_situation_columns +
                              decision_type_columns + outcome_columns)
        
        # Only keep columns that exist in the dataframe
        available_columns = [col for col in all_desired_columns if col in df.columns]
        
        print(f"📋 Selected {len(available_columns)} columns for final dataset")
        
        return df[available_columns]

    def final_quality_check(self, df):
        """
        Perform final data quality checks
        """
        print("\n🔍 Performing final quality checks...")
        
        # Check for missing values in critical columns
        critical_columns = ['Down', 'YardsToGo', 'YardLineFixed', 'OffenseTeam']
        for col in critical_columns:
            if col in df.columns:
                missing = df[col].isna().sum()
                if missing > 0:
                    print(f"⚠️  {col}: {missing} missing values")
                else:
                    print(f"✓ {col}: No missing values")
        
        # Check decision type distribution
        if 'DecisionType' in df.columns:
            decision_counts = df['DecisionType'].value_counts()
            print(f"\n📊 Decision Type Distribution:")
            for decision, count in decision_counts.items():
                print(f"   {decision}: {count} plays ({count/len(df)*100:.1f}%)")
        
        # Check conversion success rate (for go-for-it attempts)
        if 'ConversionSuccess' in df.columns:
            go_for_it_mask = df['IsGoForIt'] == 1
            successful_conversions = df.loc[go_for_it_mask, 'ConversionSuccess'].sum()
            total_attempts = go_for_it_mask.sum()
            if total_attempts > 0:
                success_rate = successful_conversions / total_attempts
                print(f"🎯 Conversion Success Rate: {successful_conversions}/{total_attempts} ({success_rate:.1%})")
        
        print("✅ Data quality checks completed")
        
        return df

    def save_processed_data(self, df, output_path="fourth_down_processed.csv"):
        """
        Save the processed data to CSV
        """
        try:
            df.to_csv(output_path, index=False)
            print(f"\n💾 Processed data saved to: {output_path}")
            print(f"📁 File contains {len(df)} fourth down plays with {len(df.columns)} features")
            
            # Print file size
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"📏 File size: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"❌ Error saving file: {e}")

    def run_complete_processing(self, data_path=".", years=range(2019, 2025), output_file="fourth_down_processed.csv"):
        """
        Run the complete data processing pipeline
        """
        print("🚀 Starting Complete 4th Down Data Processing Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Load data
            df_raw = self.load_multiple_seasons(data_path, years)
            
            # Step 2: Clean and process data
            df_processed = self.clean_and_process_data(df_raw)
            
            # Step 3: Save processed data
            self.save_processed_data(df_processed, output_file)
            
            # Step 4: Store for further analysis
            self.processed_data = df_processed
            
            print("\n" + "=" * 60)
            print("✅ Data Processing Complete!")
            print("=" * 60)
            
            return df_processed
            
        except Exception as e:
            print(f"❌ Error in processing pipeline: {e}")
            import traceback
            traceback.print_exc()

# Usage example
def main():
    """
    Main function to run the data processing
    """
    processor = FourthDownDataProcessor()
    
    # Process data from 2019-2024
    processed_data = processor.run_complete_processing(
        data_path=".",  # Current directory - adjust if needed
        years=range(2019, 2025),
        output_file="fourth_down_processed_2019_2024.csv"
    )
    
    # Display sample of processed data
    if processed_data is not None:
        print("\n📋 Sample of processed data:")
        print(processed_data.head(10))
        print(f"\n📊 Final dataset shape: {processed_data.shape}")

if __name__ == "__main__":
    main()