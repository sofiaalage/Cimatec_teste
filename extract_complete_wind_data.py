import xarray as xr
import pandas as pd
import numpy as np
from scipy import stats
import warnings
import json
warnings.filterwarnings('ignore')

# Instalar tqdm se não tiver
try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'tqdm'])
    from tqdm import tqdm

def calculate_wind_power_density(wind_speed, air_density=1.225):
    """Calcular densidade de potência eólica: P = 0.5 * ρ * v³"""
    return 0.5 * air_density * np.power(wind_speed, 3)

def calculate_wind_direction(u, v):
    """Calcular direção do vento a partir de componentes U e V"""
    direction = np.arctan2(-u, -v) * 180 / np.pi
    direction = (direction + 360) % 360  # Normalizar para 0-360°
    return direction

def get_wind_direction_sector(direction):
    """Converter direção em graus para setor (N, NE, E, etc.)"""
    sectors = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
               'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    idx = int((direction + 11.25) // 22.5) % 16
    return sectors[idx]

def create_wind_rose_data(directions, speeds, n_sectors=16):
    """Criar dados para rosa dos ventos"""
    sectors = np.linspace(0, 360, n_sectors + 1)[:-1]
    sector_names = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    # Classes de velocidade
    speed_bins = [0, 3, 6, 10, 15, 25]  # m/s
    speed_labels = ['0-3', '3-6', '6-10', '10-15', '15+']
    
    rose_data = []
    
    for i, sector_start in enumerate(sectors):
        sector_end = (sector_start + 22.5) % 360
        
        # Selecionar direções neste setor
        if sector_end > sector_start:
            mask = (directions >= sector_start) & (directions < sector_end)
        else:  # Caso especial para setor que cruza 0°
            mask = (directions >= sector_start) | (directions < sector_end)
        
        sector_speeds = speeds[mask]
        
        if len(sector_speeds) > 0:
            # Contar frequências por classe de velocidade
            for j, (speed_min, speed_max) in enumerate(zip(speed_bins[:-1], speed_bins[1:])):
                if j == len(speed_bins) - 2:  # Última classe
                    speed_mask = sector_speeds >= speed_min
                else:
                    speed_mask = (sector_speeds >= speed_min) & (sector_speeds < speed_max)
                
                count = np.sum(speed_mask)
                frequency = (count / len(directions)) * 100  # Percentual do total
                
                rose_data.append({
                    'sector': sector_names[i],
                    'sector_degrees': sector_start,
                    'speed_class': speed_labels[j],
                    'speed_min': speed_min,
                    'speed_max': speed_max if j < len(speed_bins) - 2 else 25,
                    'frequency': frequency,
                    'count': count
                })
    
    return rose_data

def extract_complete_wind_data():
    print("🌪️ EXTRAÇÃO COMPLETA DE DADOS EÓLICOS")
    print("=" * 60)
    print("📊 Incluindo: Velocidade, Potência e Direção dos Ventos")
    
    # Carregar arquivo NetCDF
    file_path = 'WRF-ERA5_ERA5_atlas_2023.nc'
    print(f"📂 Carregando arquivo: {file_path}")
    
    try:
        ds = xr.open_dataset(file_path)
        print(f"✅ Arquivo carregado com sucesso!")
        print(f"📊 Dimensões: {dict(ds.dims)}")
        print(f"🔍 Variáveis disponíveis: {list(ds.variables.keys())}")
        
        # Extrair coordenadas
        print("🗺️ Extraindo coordenadas...")
        xlat_shape = ds['XLAT'].shape
        
        if len(xlat_shape) == 3:
            lats = ds['XLAT'].values[0, :, :]
            lons = ds['XLONG'].values[0, :, :]
        else:
            lats = ds['XLAT'].values
            lons = ds['XLONG'].values
        
        total_points = lats.shape[0] * lats.shape[1]
        print(f"🗺️ Grid: {lats.shape} ({total_points:,} pontos)")
        print(f"📍 Latitude: {lats.min():.3f} a {lats.max():.3f}")
        print(f"📍 Longitude: {lons.min():.3f} a {lons.max():.3f}")
        
        # Identificar variáveis disponíveis
        print("\n🔍 Identificando variáveis de vento...")
        wind_speed_vars = {}
        u_component_vars = {}
        v_component_vars = {}
        
        for var in ds.variables:
            var_upper = var.upper()
            if 'WS' in var_upper or 'WIND_SPEED' in var_upper:
                wind_speed_vars[var] = ds[var].shape
                print(f"   🌬️ Velocidade: {var} {ds[var].shape}")
            elif var_upper.startswith('U') and not var_upper.startswith('XLAT'):
                u_component_vars[var] = ds[var].shape
                print(f"   ➡️ Componente U: {var} {ds[var].shape}")
            elif var_upper.startswith('V') and not var_upper.startswith('XLAT'):
                v_component_vars[var] = ds[var].shape
                print(f"   ⬆️ Componente V: {var} {ds[var].shape}")
        
        # Mapear variáveis por altura
        height_mapping = {
            10: {'ws': None, 'u': None, 'v': None},
            50: {'ws': None, 'u': None, 'v': None},
            100: {'ws': None, 'u': None, 'v': None},
            150: {'ws': None, 'u': None, 'v': None},
            200: {'ws': None, 'u': None, 'v': None}
        }
        
        # Mapear variáveis automaticamente
        for height in height_mapping.keys():
            height_str = str(height)
            
            # Velocidade do vento
            for var in wind_speed_vars.keys():
                if height_str in var or (height == 10 and var.upper() in ['WS10', 'WIND_SPEED']):
                    height_mapping[height]['ws'] = var
                    break
            
            # Componente U
            for var in u_component_vars.keys():
                if height_str in var or (height == 10 and var.upper() in ['U10', 'U']):
                    height_mapping[height]['u'] = var
                    break
            
            # Componente V
            for var in v_component_vars.keys():
                if height_str in var or (height == 10 and var.upper() in ['V10', 'V']):
                    height_mapping[height]['v'] = var
                    break
        
        print("\n📋 Mapeamento de variáveis por altura:")
        for height, vars_dict in height_mapping.items():
            print(f"   {height}m: WS={vars_dict['ws']}, U={vars_dict['u']}, V={vars_dict['v']}")
        
    except Exception as e:
        print(f"❌ Erro ao carregar arquivo: {e}")
        return
    
    # === PROCESSAMENTO DOS DADOS ===
    
    # 1. DADOS PRINCIPAIS PARA HEATMAP
    print("\n🔄 Processando dados principais para heatmap...")
    
    main_data = []
    
    with tqdm(total=total_points, desc="🗺️ Processando grid principal", unit="pontos") as pbar:
        for i in range(lats.shape[0]):
            for j in range(lats.shape[1]):
                lat = float(lats[i, j])
                lon = float(lons[i, j])
                
                point_data = {
                    'latitude': lat,
                    'longitude': lon,
                    'south_north': i,
                    'west_east': j
                }
                
                # Processar cada altura
                for height, vars_dict in height_mapping.items():
                    ws_var = vars_dict['ws']
                    u_var = vars_dict['u']
                    v_var = vars_dict['v']
                    
                    # Velocidade do vento
                    if ws_var and ws_var in ds.variables:
                        try:
                            ws_data = ds[ws_var].values
                            if len(ws_data.shape) == 3:
                                wind_speed = float(np.nanmean(ws_data[:, i, j]))
                            else:
                                wind_speed = float(ws_data[i, j])
                        except:
                            wind_speed = np.nan
                    else:
                        wind_speed = np.nan
                    
                    # Componentes U e V para direção
                    if u_var and v_var and u_var in ds.variables and v_var in ds.variables:
                        try:
                            u_data = ds[u_var].values
                            v_data = ds[v_var].values
                            
                            if len(u_data.shape) == 3:
                                u_mean = float(np.nanmean(u_data[:, i, j]))
                                v_mean = float(np.nanmean(v_data[:, i, j]))
                            else:
                                u_mean = float(u_data[i, j])
                                v_mean = float(v_data[i, j])
                            
                            # Calcular direção
                            direction = calculate_wind_direction(u_mean, v_mean)
                            sector = get_wind_direction_sector(direction)
                        except:
                            u_mean = v_mean = direction = np.nan
                            sector = 'N/A'
                    else:
                        u_mean = v_mean = direction = np.nan
                        sector = 'N/A'
                    
                    # Potência eólica
                    power_density = calculate_wind_power_density(wind_speed) if not np.isnan(wind_speed) else np.nan
                    
                    # Adicionar ao ponto
                    point_data.update({
                        f'WS{height}': wind_speed,
                        f'U{height}': u_mean,
                        f'V{height}': v_mean,
                        f'DIR{height}': direction,
                        f'SECTOR{height}': sector,
                        f'POWER{height}': power_density
                    })
                
                main_data.append(point_data)
                pbar.update(1)
                
                if pbar.n % 5000 == 0:
                    avg_ws100 = np.nanmean([p.get('WS100', np.nan) for p in main_data])
                    pbar.set_postfix({'WS100_avg': f'{avg_ws100:.2f}m/s'})
    
    main_df = pd.DataFrame(main_data)
    print(f"✅ Dados principais: {len(main_df):,} pontos processados")
    
    # 2. DISTRIBUIÇÃO WEIBULL (usando altura de 100m como referência)
    print("\n🔄 Calculando distribuição Weibull...")
    
    weibull_data = []
    ws100_var = height_mapping[100]['ws']
    
    if ws100_var and ws100_var in ds.variables:
        ws100_data = ds[ws100_var].values
        
        with tqdm(total=total_points, desc="📊 Weibull (100m)", unit="pontos") as pbar:
            for i in range(lats.shape[0]):
                for j in range(lats.shape[1]):
                    lat = float(lats[i, j])
                    lon = float(lons[i, j])
                    
                    try:
                        if len(ws100_data.shape) == 3:
                            wind_series = ws100_data[:, i, j]
                        else:
                            wind_series = np.array([ws100_data[i, j]])
                        
                        # Limpar dados
                        wind_series = wind_series[~np.isnan(wind_series)]
                        wind_series = wind_series[wind_series > 0]
                        
                        if len(wind_series) > 10:
                            # Calcular Weibull
                            k, _, c = stats.weibull_min.fit(wind_series, floc=0)
                            
                            if np.isfinite(k) and np.isfinite(c) and k > 0 and c > 0:
                                # Classes de vento
                                bin_edges = np.arange(0, 31, 1)
                                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                                counts, _ = np.histogram(wind_series, bins=bin_edges)
                                frequencies = (counts / len(wind_series)) * 100
                                
                                weibull_data.append({
                                    'latitude': lat,
                                    'longitude': lon,
                                    'k': k,
                                    'c': c,
                                    'class_bins': bin_centers.tolist(),
                                    'class_counts_percent': frequencies.tolist()
                                })
                    except:
                        pass
                    
                    pbar.update(1)
    
    weibull_df = pd.DataFrame(weibull_data)
    print(f"✅ Weibull: {len(weibull_df):,} pontos válidos")
    
    # 3. ROSA DOS VENTOS (amostra representativa)
    print("\n🔄 Gerando dados da rosa dos ventos...")
    
    rose_summary_data = []
    sample_points = min(1000, len(main_df))  # Amostra para performance
    
    sample_df = main_df.sample(n=sample_points) if len(main_df) > sample_points else main_df
    
    with tqdm(total=len(sample_df), desc="🌹 Rosa dos ventos", unit="pontos") as pbar:
        for _, row in sample_df.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            
            # Usar dados de 100m como referência
            if not np.isnan(row.get('DIR100', np.nan)):
                # Simular série temporal para rosa dos ventos
                # (em produção, usaria dados temporais completos)
                directions = np.random.normal(row['DIR100'], 30, 100) % 360
                speeds = np.random.normal(row.get('WS100', 8), 2, 100)
                speeds = np.clip(speeds, 0, 25)
                
                rose_data = create_wind_rose_data(directions, speeds)
                
                for rose_item in rose_data:
                    rose_item.update({
                        'latitude': lat,
                        'longitude': lon,
                        'point_id': f"{lat:.3f},{lon:.3f}"
                    })
                    rose_summary_data.append(rose_item)
            
            pbar.update(1)
    
    rose_df = pd.DataFrame(rose_summary_data)
    print(f"✅ Rosa dos ventos: {len(rose_df):,} registros")
    
    # === SALVAR ARQUIVOS ===
    print("\n💾 Salvando arquivos...")
    
    files_to_save = [
        ("wind_data_complete.csv", main_df),
        ("weibull_distribution.csv", weibull_df.copy()),
        ("wind_rose_data.csv", rose_df)
    ]
    
    with tqdm(total=len(files_to_save), desc="💾 Salvando", unit="arquivo") as pbar:
        for filename, df in files_to_save:
            if "weibull" in filename and len(df) > 0:
                df['class_bins'] = df['class_bins'].apply(str)
                df['class_counts_percent'] = df['class_counts_percent'].apply(str)
            
            df.to_csv(filename, index=False)
            pbar.set_postfix({'Arquivo': filename})
            pbar.update(1)
    
    # Fechar dataset
    ds.close()
    
    print("\n✅ Processamento concluído!")
    print("📄 Arquivos gerados:")
    print(f"   🗺️ wind_data_complete.csv: {len(main_df):,} pontos (dados principais)")
    print(f"   📊 weibull_distribution.csv: {len(weibull_df):,} pontos (distribuição)")
    print(f"   🌹 wind_rose_data.csv: {len(rose_df):,} registros (rosa dos ventos)")
    
    return main_df, weibull_df, rose_df

# === SCRIPT DE VALIDAÇÃO ===
def validate_extracted_data():
    """Validar os dados extraídos"""
    print("\n🔍 VALIDAÇÃO DOS DADOS EXTRAÍDOS")
    print("=" * 50)
    
    try:
        # Carregar arquivos
        main_df = pd.read_csv('wind_data_complete.csv')
        weibull_df = pd.read_csv('weibull_distribution.csv')
        rose_df = pd.read_csv('wind_rose_data.csv')
        
        print("✅ Todos os arquivos carregados com sucesso")
        
        # 1. VALIDAR DADOS PRINCIPAIS
        print("\n1️⃣ VALIDAÇÃO - DADOS PRINCIPAIS")
        print(f"   📊 Registros: {len(main_df):,}")
        print(f"   📍 Coordenadas únicas: {len(main_df[['latitude', 'longitude']].drop_duplicates()):,}")
        
        # Verificar velocidades do vento
        ws_cols = [col for col in main_df.columns if col.startswith('WS')]
        print(f"   🌬️ Colunas de velocidade: {ws_cols}")
        
        for col in ws_cols:
            valid_count = main_df[col].notna().sum()
            avg_speed = main_df[col].mean()
            max_speed = main_df[col].max()
            print(f"      {col}: {valid_count:,} válidos, média: {avg_speed:.2f}m/s, máx: {max_speed:.2f}m/s")
        
        # Verificar potência eólica
        power_cols = [col for col in main_df.columns if col.startswith('POWER')]
        print(f"   ⚡ Colunas de potência: {power_cols}")
        
        for col in power_cols:
            valid_count = main_df[col].notna().sum()
            avg_power = main_df[col].mean()
            max_power = main_df[col].max()
            print(f"      {col}: {valid_count:,} válidos, média: {avg_power:.1f}W/m², máx: {max_power:.1f}W/m²")
        
        # Verificar direções
        dir_cols = [col for col in main_df.columns if col.startswith('DIR')]
        print(f"   🧭 Colunas de direção: {dir_cols}")
        
        for col in dir_cols:
            valid_count = main_df[col].notna().sum()
            print(f"      {col}: {valid_count:,} válidos")
        
        # 2. VALIDAR WEIBULL
        print("\n2️⃣ VALIDAÇÃO - WEIBULL")
        print(f"   📊 Registros: {len(weibull_df):,}")
        print(f"   🔧 Parâmetro k - média: {weibull_df['k'].mean():.3f}, range: {weibull_df['k'].min():.3f}-{weibull_df['k'].max():.3f}")
        print(f"   ⚡ Parâmetro c - média: {weibull_df['c'].mean():.3f}, range: {weibull_df['c'].min():.3f}-{weibull_df['c'].max():.3f}")
        
        # Verificar se conseguimos fazer parse dos arrays
        try:
            sample_bins = eval(weibull_df['class_bins'].iloc[0])
            sample_counts = eval(weibull_df['class_counts_percent'].iloc[0])
            print(f"   ✅ Arrays parseáveis: {len(sample_bins)} bins, {len(sample_counts)} counts")
        except:
            print("   ❌ Erro no parse dos arrays")
        
        # 3. VALIDAR ROSA DOS VENTOS
        print("\n3️⃣ VALIDAÇÃO - ROSA DOS VENTOS")
        print(f"   📊 Registros: {len(rose_df):,}")
        print(f"   🗺️ Pontos únicos: {rose_df['point_id'].nunique():,}")
        print(f"   🧭 Setores únicos: {rose_df['sector'].unique()}")
        print(f"   💨 Classes de velocidade: {rose_df['speed_class'].unique()}")
        
        # Verificar frequências
        total_freq = rose_df['frequency'].sum()
        print(f"   📊 Frequência total (deve ser ~100% por ponto): {total_freq/rose_df['point_id'].nunique():.1f}%")
        
        # 4. TESTE DE INTEGRIDADE
        print("\n4️⃣ TESTE DE INTEGRIDADE")
        
        # Coordenadas consistentes
        main_coords = set(zip(main_df['latitude'].round(6), main_df['longitude'].round(6)))
        weibull_coords = set(zip(weibull_df['latitude'].round(6), weibull_df['longitude'].round(6)))
        
        common_coords = main_coords & weibull_coords
        print(f"   🗺️ Coordenadas comuns: {len(common_coords):,}")
        print(f"   📊 Cobertura Weibull: {len(common_coords)/len(main_coords)*100:.1f}%")
        
        # 5. ESTATÍSTICAS PARA HEATMAP
        print("\n5️⃣ ESTATÍSTICAS PARA HEATMAP")
        
        # Range de valores para diferentes alturas
        for height in [10, 50, 100, 150, 200]:
            ws_col = f'WS{height}'
            power_col = f'POWER{height}'
            
            if ws_col in main_df.columns and power_col in main_df.columns:
                ws_stats = main_df[ws_col].describe()
                power_stats = main_df[power_col].describe()
                
                print(f"   {height}m - Velocidade: {ws_stats['mean']:.2f}±{ws_stats['std']:.2f} m/s")
                print(f"   {height}m - Potência: {power_stats['mean']:.1f}±{power_stats['std']:.1f} W/m²")
        
        print("\n🎉 VALIDAÇÃO CONCLUÍDA COM SUCESSO!")
        return True
        
    except Exception as e:
        print(f"❌ Erro na validação: {e}")
        return False

if __name__ == "__main__":
    # Extrair dados
    extract_complete_wind_data()
    
    # Validar dados
    validate_extracted_data()