import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# .env 파일 로드
load_dotenv()

@dataclass
class WeatherData:
    """날씨 데이터 구조체 - Main Agent에서 상태/설명 처리용"""
    date: str
    location: str
    temperature_min: float
    temperature_max: float
    precipitation_probability: int
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    data_source: str = 'unknown'  # 'short_forecast', 'medium_forecast', 'statistical_prediction'
    confidence: float = 0.0  # 0.0 ~ 1.0
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return asdict(self)

@dataclass
class WeatherResponse:
    """Main Agent로 전달할 응답 구조체"""
    success: bool
    data: Optional[WeatherData]
    error_message: Optional[str]
    request_info: Dict
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'success': self.success,
            'data': self.data.to_dict() if self.data else None,
            'error_message': self.error_message,
            'request_info': self.request_info
        }

class WeatherAgent:
    """날씨 에이전트 - 기온과 강수확률만 예측"""
    
    def __init__(self, historical_data_path: str = None):
        # 환경변수에서 API 키 로드
        self.kma_short_api_key = os.getenv('KMA_SHORT_FORECAST_API_KEY')
        self.kma_medium_api_key = os.getenv('KMA_MEDIUM_FORECAST_API_KEY')
        
        # 과거 데이터 경로
        self.historical_data_path = historical_data_path or os.getenv('HISTORICAL_DATA_PATH', 'data/weather_history.csv')
        
        # 통계 데이터 (장기 예보용)
        self.historical_stats = None
        
        # 초기화 상태
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """에이전트 초기화"""
        try:
            # API 키 검증
            if not all([self.kma_short_api_key, self.kma_medium_api_key]):
                raise ValueError("Missing required API keys in .env file")
            
            # 과거 통계 데이터 로드
            if not self._load_historical_statistics():
                raise ValueError("Failed to load historical statistics")
                
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Weather Agent initialization failed: {e}")
            return False
    
    def get_weather_forecast(self, location: str, target_date: str) -> WeatherResponse:
        """
        메인 날씨 조회 함수 - 단일 날짜 조회
        
        Args:
            location: 위치 (예: "홍대", "서울특별시 마포구")
            target_date: 목표 날짜 (YYYY-MM-DD)
            
        Returns:
            WeatherResponse: 결과 데이터
        """
        request_info = {
            'location': location,
            'target_date': target_date,
            'timestamp': datetime.now().isoformat()
        }
        
        if not self.is_initialized:
            return WeatherResponse(
                success=False,
                data=None,
                error_message="Weather Agent not initialized",
                request_info=request_info
            )
        
        try:
            # [Step 2] 날짜 및 지역 파싱
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            lat, lon = self._geocode_location(location)
            
            if lat is None or lon is None:
                return WeatherResponse(
                    success=False,
                    data=None,
                    error_message=f"Failed to geocode location: {location}",
                    request_info=request_info
                )
            
            # [Step 3] 오늘 날짜와 비교하여 '며칠 후' 계산
            today = datetime.now().date()
            target_date_obj = target_dt.date()
            delta_days = (target_date_obj - today).days
            
            request_info['delta_days'] = delta_days
            
            # [Step 4] 예보 구간 판단 및 분기
            weather_data = None
            
            if 0 <= delta_days <= 3:
                # 단기 예보 API
                weather_data = self._get_short_forecast_data(lat, lon, location, target_date, delta_days)
            elif 4 <= delta_days <= 10:
                # 중기 예보 API  
                weather_data = self._get_medium_forecast_data(lat, lon, location, target_date, delta_days)
            else:
                # 과거 날씨 통계 기반 추정
                weather_data = self._get_statistical_prediction(lat, lon, location, target_date, target_dt)
            
            if weather_data:
                return WeatherResponse(
                    success=True,
                    data=weather_data,
                    error_message=None,
                    request_info=request_info
                )
            else:
                return WeatherResponse(
                    success=False,
                    data=None,
                    error_message="Failed to retrieve weather data",
                    request_info=request_info
                )
            
        except Exception as e:
            return WeatherResponse(
                success=False,
                data=None,
                error_message=str(e),
                request_info=request_info
            )
    
    def _get_short_forecast_data(self, lat: float, lon: float, location: str, 
                               target_date: str, delta_days: int) -> Optional[WeatherData]:
        """[Step 5-6] 단기예보 API 호출 및 target_date 정보만 추출"""
        try:
            # 기상청 단기예보 API 호출
            base_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
            nx, ny = self._convert_to_kma_grid(lat, lon)
            
            base_date = datetime.now().strftime('%Y%m%d')
            base_time = self._get_base_time()
            
            params = {
                'serviceKey': self.kma_short_api_key,
                'numOfRows': '1000',
                'pageNo': '1',
                'dataType': 'JSON',
                'base_date': base_date,
                'base_time': base_time,
                'nx': nx,
                'ny': ny
            }
            
            response = requests.get(base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                api_data = response.json()
                # [Step 6] target_date에 해당하는 값만 필터링 및 추출
                parsed_data = self._parse_short_forecast_for_date(api_data, target_date)
                
                if parsed_data:
                    return WeatherData(
                        date=target_date,
                        location=location,
                        temperature_min=parsed_data.get('temp_min', 0),
                        temperature_max=parsed_data.get('temp_max', 0),
                        precipitation_probability=parsed_data.get('precip_prob', 0),
                        humidity=parsed_data.get('humidity'),
                        wind_speed=parsed_data.get('wind_speed'),
                        data_source='short_forecast',
                        confidence=0.9
                    )
            
            return None
            
        except Exception as e:
            print(f"Short forecast API error: {e}")
            return None
    
    def _get_medium_forecast_data(self, lat: float, lon: float, location: str, 
                                target_date: str, delta_days: int) -> Optional[WeatherData]:
        """[Step 5-6] 중기예보 API 호출 및 target_date 정보만 추출"""
        try:
            # 기상청 중기예보 API 호출
            base_url = "http://apis.data.go.kr/1360000/MidFcstInfoService/getMidFcst"
            region_id = self._get_region_id(lat, lon)
            
            params = {
                'serviceKey': self.kma_medium_api_key,
                'numOfRows': '10',
                'pageNo': '1',
                'dataType': 'JSON',
                'regId': region_id,
                'tmFc': datetime.now().strftime('%Y%m%d') + '0600'
            }
            
            response = requests.get(base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                api_data = response.json()
                # [Step 6] target_date에 해당하는 값만 필터링 및 추출
                parsed_data = self._parse_medium_forecast_for_date(api_data, target_date, delta_days)
                
                if parsed_data:
                    return WeatherData(
                        date=target_date,
                        location=location,
                        temperature_min=parsed_data.get('temp_min', 0),
                        temperature_max=parsed_data.get('temp_max', 0),
                        precipitation_probability=parsed_data.get('precip_prob', 0),
                        humidity=parsed_data.get('humidity'),
                        wind_speed=parsed_data.get('wind_speed'),
                        data_source='medium_forecast',
                        confidence=0.8
                    )
            
            return None
            
        except Exception as e:
            print(f"Medium forecast API error: {e}")
            return None
    
    def _get_statistical_prediction(self, lat: float, lon: float, location: str, 
                                  target_date: str, target_dt: datetime) -> Optional[WeatherData]:
        """[Step 5-6] 과거 날씨 통계 기반 추정 (delta_days > 10)"""
        try:
            if self.historical_stats is None:
                return None
            
            # 해당 지역의 과거 동일 시기 통계 조회
            month = target_dt.month
            day = target_dt.day
            
            # 지역별 통계 데이터 필터링
            location_stats = self._get_location_statistics(location, lat, lon)
            
            # 해당 월/일의 과거 평균 계산
            date_stats = self._get_date_statistics(location_stats, month, day)
            
            if date_stats:
                return WeatherData(
                    date=target_date,
                    location=location,
                    temperature_min=round(date_stats.get('temp_min_avg', 10), 1),
                    temperature_max=round(date_stats.get('temp_max_avg', 20), 1),
                    precipitation_probability=int(date_stats.get('precip_prob_avg', 30)),
                    humidity=round(date_stats.get('humidity_avg', 60), 1) if date_stats.get('humidity_avg') else None,
                    wind_speed=round(date_stats.get('wind_speed_avg', 3), 1) if date_stats.get('wind_speed_avg') else None,
                    data_source='statistical_prediction',
                    confidence=0.6
                )
            
            return None
            
        except Exception as e:
            print(f"Statistical prediction error: {e}")
            return None
    
    def _load_historical_statistics(self) -> bool:
        """과거 날씨 통계 데이터 CSV 로드"""
        try:
            if not os.path.exists(self.historical_data_path):
                print(f"Historical data file not found: {self.historical_data_path}")
                return False
            
            # CSV 데이터 로드
            df = pd.read_csv(self.historical_data_path)
            
            # 필수 컬럼 확인
            required_columns = ['date', 'location', 'temp_min', 'temp_max', 'precip_prob']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Missing required columns: {missing_columns}")
                return False
            
            # 날짜 변환
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            
            # 통계 데이터로 저장
            self.historical_stats = df
            
            print(f"Historical statistics loaded: {len(df)} records")
            return True
            
        except Exception as e:
            print(f"Error loading historical statistics: {e}")
            return False
    
    def _parse_short_forecast_for_date(self, api_response: Dict, target_date: str) -> Optional[Dict]:
        """단기예보 API 응답에서 target_date 데이터만 파싱"""
        try:
            items = api_response.get('response', {}).get('body', {}).get('items', {}).get('item', [])
            
            target_date_str = target_date.replace('-', '')  # YYYYMMDD 형식으로 변환
            weather_data = {}
            temp_values = []
            
            for item in items:
                if item.get('fcstDate') == target_date_str:
                    category = item.get('category')
                    value = item.get('fcstValue')
                    time = item.get('fcstTime', '0000')
                    
                    if category == 'TMP':  # 기온
                        temp_values.append(float(value))
                    elif category == 'REH':  # 습도
                        weather_data['humidity'] = float(value)
                    elif category == 'POP':  # 강수확률
                        weather_data['precip_prob'] = int(value)
                    elif category == 'WSD':  # 풍속
                        weather_data['wind_speed'] = float(value)
            
            # 최고/최저 기온 계산
            if temp_values:
                weather_data['temp_min'] = min(temp_values)
                weather_data['temp_max'] = max(temp_values)
            
            return weather_data if weather_data else None
            
        except Exception as e:
            print(f"Short forecast parsing error: {e}")
            return None
    
    def _parse_medium_forecast_for_date(self, api_response: Dict, target_date: str, delta_days: int) -> Optional[Dict]:
        """중기예보 API 응답에서 target_date 데이터만 파싱"""
        try:
            items = api_response.get('response', {}).get('body', {}).get('items', {}).get('item', [])
            
            if not items:
                return None
            
            # 중기예보는 3일후부터 시작하므로 인덱스 계산
            day_index = delta_days - 3  # 4일후 -> index 1, 5일후 -> index 2
            
            item = items[0] if items else {}
            
            # 중기예보 필드명 (예시)
            temp_min_field = f'taMin{day_index + 3}'  # taMin3, taMin4, ...
            temp_max_field = f'taMax{day_index + 3}'  # taMax3, taMax4, ...
            precip_prob_field = f'rnSt{day_index + 3}'  # rnSt3, rnSt4, ...
            
            weather_data = {}
            
            if temp_min_field in item:
                weather_data['temp_min'] = float(item[temp_min_field])
            if temp_max_field in item:
                weather_data['temp_max'] = float(item[temp_max_field])
            if precip_prob_field in item:
                weather_data['precip_prob'] = int(item[precip_prob_field])
            
            return weather_data if weather_data else None
            
        except Exception as e:
            print(f"Medium forecast parsing error: {e}")
            return None
    
    def _get_location_statistics(self, location: str, lat: float, lon: float) -> pd.DataFrame:
        """지역별 통계 데이터 필터링"""
        if self.historical_stats is None:
            return pd.DataFrame()
        
        # 정확한 지역명 매칭 시도
        location_data = self.historical_stats[self.historical_stats['location'] == location]
        
        if len(location_data) > 0:
            return location_data
        
        # 부분 매칭 시도
        for loc in self.historical_stats['location'].unique():
            if location in loc or loc in location:
                return self.historical_stats[self.historical_stats['location'] == loc]
        
        # 가장 가까운 지역의 데이터 사용 (좌표 기반)
        # 실제로는 더 정교한 지역 매칭 로직 필요
        return self.historical_stats.head(1000)  # 기본값
    
    def _get_date_statistics(self, location_data: pd.DataFrame, month: int, day: int) -> Optional[Dict]:
        """특정 월/일의 과거 통계 계산"""
        try:
            # 해당 월/일 데이터 필터링
            date_data = location_data[
                (location_data['month'] == month) & 
                (location_data['day'] == day)
            ]
            
            if len(date_data) == 0:
                # 해당 월 전체 데이터로 대체
                date_data = location_data[location_data['month'] == month]
            
            if len(date_data) == 0:
                return None
            
            # 통계 계산
            stats = {
                'temp_min_avg': date_data['temp_min'].mean(),
                'temp_max_avg': date_data['temp_max'].mean(),
                'precip_prob_avg': date_data['precip_prob'].mean()
            }
            
            # 선택적 컬럼들
            if 'humidity' in date_data.columns:
                stats['humidity_avg'] = date_data['humidity'].mean()
            if 'wind_speed' in date_data.columns:
                stats['wind_speed_avg'] = date_data['wind_speed'].mean()
            
            return stats
            
        except Exception as e:
            print(f"Date statistics calculation error: {e}")
            return None
    
    # 헬퍼 메서드들
    def _convert_to_kma_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """위경도를 기상청 격자좌표로 변환"""
        # 기상청 Lambert Conformal Conic 투영 변환 공식
        # 실제 구현에서는 정확한 변환 공식 사용
        
        # 간단한 근사 변환 (실제로는 더 정확한 공식 필요)
        nx = int((lon - 124.0) * 20 + 1)
        ny = int((lat - 33.0) * 20 + 1)
        
        # 격자 범위 제한
        nx = max(1, min(149, nx))
        ny = max(1, min(253, ny))
        
        return nx, ny
    
    def _get_base_time(self) -> str:
        """기상청 API 호출용 기준시간 계산"""
        now = datetime.now()
        base_times = ['0200', '0500', '0800', '1100', '1400', '1700', '2000', '2300']
        
        current_hour = now.hour
        
        # 현재 시간 이전의 가장 최근 기준시간 찾기
        for i in range(len(base_times) - 1, -1, -1):
            bt_hour = int(base_times[i][:2])
            if current_hour >= bt_hour:
                return base_times[i]
        
        # 자정 이전이면 전날 마지막 기준시간
        return base_times[-1]
    
    def _get_region_id(self, lat: float, lon: float) -> str:
        """중기예보 지역코드 반환"""
        # 주요 지역별 코드 매핑
        region_codes = {
            '11B00000': {'name': '서울/경기', 'lat_range': (37.0, 38.0), 'lon_range': (126.5, 127.5)},
            '11D10000': {'name': '강원영서', 'lat_range': (37.5, 38.5), 'lon_range': (127.5, 128.5)},
            '11C20000': {'name': '대전/충남', 'lat_range': (36.0, 37.0), 'lon_range': (126.5, 127.5)},
            '11F20000': {'name': '광주/전남', 'lat_range': (34.5, 35.5), 'lon_range': (126.0, 127.0)},
            '11H20000': {'name': '부산/경남', 'lat_range': (35.0, 36.0), 'lon_range': (128.5, 129.5)},
        }
        
        # 좌표에 맞는 지역코드 찾기
        for code, info in region_codes.items():
            lat_min, lat_max = info['lat_range']
            lon_min, lon_max = info['lon_range']
            
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return code
        
        return '11B00000'  # 기본값 (서울/경기)
    
    def _geocode_location(self, location: str) -> Tuple[Optional[float], Optional[float]]:
        """위치명을 좌표로 변환"""
        # 확장된 지역 좌표 매핑
        location_map = {
            # 서울 지역
            '서울': (37.5665, 126.9780), '서울특별시': (37.5665, 126.9780),
            '홍대': (37.5566, 126.9240), '강남': (37.5173, 127.0473),
            '명동': (37.5636, 126.9804), '이태원': (37.5343, 126.9948),
            '신촌': (37.5559, 126.9365), '종로': (37.5735, 126.9788),
            
            # 광역시
            '부산': (35.1796, 129.0756), '부산광역시': (35.1796, 129.0756),
            '대구': (35.8714, 128.6014), '대구광역시': (35.8714, 128.6014),
            '인천': (37.4563, 126.7052), '인천광역시': (37.4563, 126.7052),
            '광주': (35.1595, 126.8526), '광주광역시': (35.1595, 126.8526),
            '대전': (36.3504, 127.3845), '대전광역시': (36.3504, 127.3845),
            '울산': (35.5384, 129.3114), '울산광역시': (35.5384, 129.3114),
            
            # 도청소재지
            '수원': (37.2636, 127.0286), '춘천': (37.8813, 127.7298),
            '청주': (36.6424, 127.4890), '전주': (35.8242, 127.1480),
            '포항': (36.0190, 129.3435), '창원': (35.2281, 128.6811),
            '제주': (33.4996, 126.5312), '제주시': (33.4996, 126.5312)
        }
        
        # 정확한 매칭
        if location in location_map:
            return location_map[location]
        
        # 부분 매칭
        for key, coords in location_map.items():
            if location in key or key in location:
                return coords
        
        # 매칭 실패
        print(f"Location not found: {location}")
        return None, None

# Main Agent에서 사용할 인터페이스
def create_weather_agent(historical_data_path: str = None) -> WeatherAgent:
    """Weather Agent 팩토리 함수"""
    agent = WeatherAgent(historical_data_path)
    
    if agent.initialize():
        return agent
    else:
        raise Exception("Failed to initialize Weather Agent")

# 사용 예시 (Main Agent에서)
if __name__ == "__main__":
    try:
        # [Step 1] 사용자 입력 예시
        user_input = "다음 주 화요일 홍대 날씨"
        
        # Weather Agent 생성
        weather_agent = create_weather_agent("data/weather_history_7years.csv")
        
        # [Step 2~7] 날씨 조회 요청 (실제로는 Main Agent에서 날짜/지역 파싱 후 호출)
        response = weather_agent.get_weather_forecast(
            location="홍대",
            target_date="2024-08-15"
        )
        
        # Main Agent로 전달할 응답 데이터
        if response.success:
            weather_data = response.data
            print(f"Weather forecast for {weather_data.date}:")
            print(f"  Location: {weather_data.location}")
            print(f"  Temperature: {weather_data.temperature_min}°C ~ {weather_data.temperature_max}°C")
            print(f"  Precipitation Probability: {weather_data.precipitation_probability}%")
            print(f"  Data Source: {weather_data.data_source}")
            print(f"  Confidence: {weather_data.confidence}")
        else:
            print(f"Error: {response.error_message}")
            
    except Exception as e:
        print(f"Weather Agent creation failed: {e}")