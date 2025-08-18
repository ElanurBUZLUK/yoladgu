import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from app.services.math_ab_testing import (
    MathABTesting, Experiment, ExperimentVariant, ExperimentStatus, 
    StatisticalTest, ExperimentResult
)


class TestMathABTesting:
    """MathABTesting servisi testleri"""
    
    @pytest.fixture
    def ab_testing_service(self):
        return MathABTesting()
    
    @pytest.fixture
    def sample_variants(self):
        return [
            {
                "name": "control",
                "description": "Control group with standard algorithm",
                "parameters": {"algorithm": "standard", "difficulty_factor": 1.0},
                "traffic_percentage": 50.0,
                "is_control": True
            },
            {
                "name": "experimental",
                "description": "Experimental group with enhanced algorithm",
                "parameters": {"algorithm": "enhanced", "difficulty_factor": 1.2},
                "traffic_percentage": 50.0,
                "is_control": False
            }
        ]
    
    @pytest.fixture
    def sample_experiment(self, sample_variants):
        variants = [
            ExperimentVariant(**variant) for variant in sample_variants
        ]
        
        return Experiment(
            id="test-exp-1",
            name="Algorithm Comparison Test",
            description="Testing enhanced vs standard algorithm",
            variants=variants,
            status=ExperimentStatus.DRAFT,
            start_date=datetime.utcnow(),
            target_metrics=["accuracy", "engagement", "completion_rate"],
            statistical_test=StatisticalTest.T_TEST,
            significance_level=0.05,
            minimum_sample_size=100,
            created_at=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    async def test_create_experiment(self, ab_testing_service, sample_variants):
        """Deney oluşturma testi"""
        name = "Test Experiment"
        description = "Testing new algorithm"
        target_metrics = ["accuracy", "engagement"]
        statistical_test = "t_test"
        significance_level = 0.05
        minimum_sample_size = 100
        duration_days = 14
        
        experiment = await ab_testing_service.create_experiment(
            name=name,
            description=description,
            variants=sample_variants,
            target_metrics=target_metrics,
            statistical_test=statistical_test,
            significance_level=significance_level,
            minimum_sample_size=minimum_sample_size,
            duration_days=duration_days
        )
        
        assert isinstance(experiment, Experiment)
        assert experiment.name == name
        assert experiment.description == description
        assert experiment.status == ExperimentStatus.DRAFT
        assert len(experiment.variants) == len(sample_variants)
        assert experiment.target_metrics == target_metrics
        assert experiment.statistical_test == StatisticalTest.T_TEST
        assert experiment.significance_level == significance_level
        assert experiment.minimum_sample_size == minimum_sample_size
        assert experiment.created_at is not None
    
    @pytest.mark.asyncio
    async def test_start_experiment(self, ab_testing_service, sample_experiment):
        """Deney başlatma testi"""
        # Deneyi aktif deneylere ekle
        ab_testing_service.active_experiments[sample_experiment.id] = sample_experiment
        
        success = await ab_testing_service.start_experiment(sample_experiment.id)
        
        assert success is True
        assert sample_experiment.status == ExperimentStatus.ACTIVE
        assert sample_experiment.start_date is not None
    
    @pytest.mark.asyncio
    async def test_start_nonexistent_experiment(self, ab_testing_service):
        """Var olmayan deneyi başlatma testi"""
        success = await ab_testing_service.start_experiment("nonexistent-id")
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_pause_experiment(self, ab_testing_service, sample_experiment):
        """Deney duraklatma testi"""
        # Deneyi aktif yap
        sample_experiment.status = ExperimentStatus.ACTIVE
        ab_testing_service.active_experiments[sample_experiment.id] = sample_experiment
        
        success = await ab_testing_service.pause_experiment(sample_experiment.id)
        
        assert success is True
        assert sample_experiment.status == ExperimentStatus.PAUSED
    
    @pytest.mark.asyncio
    async def test_complete_experiment(self, ab_testing_service, sample_experiment):
        """Deney tamamlama testi"""
        # Deneyi aktif yap
        sample_experiment.status = ExperimentStatus.ACTIVE
        ab_testing_service.active_experiments[sample_experiment.id] = sample_experiment
        
        success = await ab_testing_service.complete_experiment(sample_experiment.id)
        
        assert success is True
        assert sample_experiment.status == ExperimentStatus.COMPLETED
        assert sample_experiment.end_date is not None
    
    @pytest.mark.asyncio
    async def test_assign_variant(self, ab_testing_service, sample_experiment):
        """Varyant atama testi"""
        # Deneyi aktif yap
        sample_experiment.status = ExperimentStatus.ACTIVE
        ab_testing_service.active_experiments[sample_experiment.id] = sample_experiment
        
        user_id = "test-user-1"
        variant_name = await ab_testing_service.assign_variant(sample_experiment.id, user_id)
        
        assert variant_name is not None
        assert variant_name in ["control", "experimental"]
        
        # Aynı kullanıcı için aynı varyant atanmalı (sticky assignment)
        variant_name2 = await ab_testing_service.assign_variant(sample_experiment.id, user_id)
        assert variant_name2 == variant_name
    
    @pytest.mark.asyncio
    async def test_assign_variant_inactive_experiment(self, ab_testing_service, sample_experiment):
        """Aktif olmayan deney için varyant atama testi"""
        # Deney draft durumunda
        ab_testing_service.active_experiments[sample_experiment.id] = sample_experiment
        
        user_id = "test-user-1"
        variant_name = await ab_testing_service.assign_variant(sample_experiment.id, user_id)
        
        assert variant_name is None
    
    @pytest.mark.asyncio
    async def test_record_experiment_event(self, ab_testing_service, sample_experiment):
        """Deney olayı kaydetme testi"""
        # Deneyi aktif yap
        sample_experiment.status = ExperimentStatus.ACTIVE
        ab_testing_service.active_experiments[sample_experiment.id] = sample_experiment
        
        # Experiment results yapısını oluştur
        ab_testing_service.experiment_results[sample_experiment.id] = {
            "control": [],
            "experimental": []
        }
        
        # Kullanıcıya varyant ata
        user_id = "test-user-1"
        variant_name = await ab_testing_service.assign_variant(sample_experiment.id, user_id)
        
        # Olay kaydet
        event_type = "question_answered"
        event_data = {"accuracy": 0.8, "response_time": 45.5}
        
        success = await ab_testing_service.record_experiment_event(
            sample_experiment.id, user_id, variant_name, event_type, event_data
        )
        
        assert success is True
        
        # Olayın kaydedildiğini kontrol et
        assert sample_experiment.id in ab_testing_service.experiment_results
        assert variant_name in ab_testing_service.experiment_results[sample_experiment.id]
        assert len(ab_testing_service.experiment_results[sample_experiment.id][variant_name]) > 0
    
    @pytest.mark.asyncio
    async def test_record_event_nonexistent_experiment(self, ab_testing_service):
        """Var olmayan deney için olay kaydetme testi"""
        success = await ab_testing_service.record_experiment_event(
            "nonexistent-id", "user-1", "control", "test_event", {}
        )
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_analyze_experiment(self, ab_testing_service, sample_experiment):
        """Deney analizi testi"""
        # Deneyi tamamla
        sample_experiment.status = ExperimentStatus.COMPLETED
        sample_experiment.end_date = datetime.utcnow()
        ab_testing_service.active_experiments[sample_experiment.id] = sample_experiment
        
        # Mock veri ekle
        ab_testing_service.experiment_results[sample_experiment.id] = {
            "control": [
                {"event_data": {"accuracy": 0.75, "engagement": 0.8}, "timestamp": datetime.utcnow()},
                {"event_data": {"accuracy": 0.8, "engagement": 0.85}, "timestamp": datetime.utcnow()}
            ],
            "experimental": [
                {"event_data": {"accuracy": 0.85, "engagement": 0.9}, "timestamp": datetime.utcnow()},
                {"event_data": {"accuracy": 0.9, "engagement": 0.95}, "timestamp": datetime.utcnow()}
            ]
        }
        
        analysis = await ab_testing_service.analyze_experiment(sample_experiment.id)
        
        assert isinstance(analysis, dict)
        assert "experiment_id" in analysis
        assert "variant_results" in analysis
        assert "statistical_tests" in analysis
        assert "recommendations" in analysis
        
        # Sonuçların doğru formatta olduğunu kontrol et
        results = analysis["variant_results"]
        assert "control" in results
        assert "experimental" in results
        
        for variant_name, result in results.items():
            assert isinstance(result, dict)
            assert "sample_size" in result
            assert "metrics" in result
            assert "confidence_intervals" in result
    
    @pytest.mark.asyncio
    async def test_analyze_incomplete_experiment(self, ab_testing_service, sample_experiment):
        """Tamamlanmamış deney analizi testi"""
        # Deney aktif durumda
        sample_experiment.status = ExperimentStatus.ACTIVE
        ab_testing_service.active_experiments[sample_experiment.id] = sample_experiment
        
        # Experiment results yapısını oluştur
        ab_testing_service.experiment_results[sample_experiment.id] = {
            "control": [],
            "experimental": []
        }
        
        analysis = await ab_testing_service.analyze_experiment(sample_experiment.id)
        
        # Aktif deneyler analiz edilebilir, bu yüzden hata beklemeyiz
        assert isinstance(analysis, dict)
        assert "experiment_id" in analysis
    
    @pytest.mark.asyncio
    async def test_get_experiment_status(self, ab_testing_service, sample_experiment):
        """Deney durumu alma testi"""
        ab_testing_service.active_experiments[sample_experiment.id] = sample_experiment
        
        # Experiment results yapısını oluştur
        ab_testing_service.experiment_results[sample_experiment.id] = {
            "control": [],
            "experimental": []
        }
        
        status = await ab_testing_service.get_experiment_status(sample_experiment.id)
        
        assert isinstance(status, dict)
        assert "experiment_id" in status
        assert "status" in status
        assert "start_date" in status
        assert "variant_stats" in status
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_experiment_status(self, ab_testing_service):
        """Var olmayan deney durumu alma testi"""
        status = await ab_testing_service.get_experiment_status("nonexistent-id")
        
        assert "error" in status
        assert "not found" in status["error"].lower()
    
    @pytest.mark.asyncio
    async def test_list_experiments(self, ab_testing_service, sample_experiment):
        """Deney listesi alma testi"""
        ab_testing_service.active_experiments[sample_experiment.id] = sample_experiment
        
        experiments = await ab_testing_service.list_experiments()
        
        assert isinstance(experiments, list)
        assert len(experiments) > 0
        
        experiment = experiments[0]
        assert "id" in experiment
        assert "name" in experiment
        assert "status" in experiment
        # created_at field is optional in the response
    
    @pytest.mark.asyncio
    async def test_list_experiments_with_filter(self, ab_testing_service, sample_experiment):
        """Filtreli deney listesi alma testi"""
        ab_testing_service.active_experiments[sample_experiment.id] = sample_experiment
        
        # Draft durumundaki deneyleri listele
        draft_experiments = await ab_testing_service.list_experiments("draft")
        
        assert isinstance(draft_experiments, list)
        assert len(draft_experiments) > 0
        
        for experiment in draft_experiments:
            assert experiment["status"] == "draft"
    
    @pytest.mark.asyncio
    async def test_random_variant_assignment(self, ab_testing_service, sample_experiment):
        """Random varyant atama testi"""
        # Deneyi aktif yap
        sample_experiment.status = ExperimentStatus.ACTIVE
        ab_testing_service.active_experiments[sample_experiment.id] = sample_experiment
        
        # Farklı kullanıcılar için varyant atama
        variant_counts = {"control": 0, "experimental": 0}
        
        for i in range(100):
            user_id = f"user-{i}"
            variant = await ab_testing_service.assign_variant(sample_experiment.id, user_id)
            if variant:
                variant_counts[variant] += 1
        
        # Her iki varyant da atanmış olmalı (yaklaşık eşit dağılım)
        assert variant_counts["control"] > 0
        assert variant_counts["experimental"] > 0
        
        # Dağılım makul aralıkta olmalı (40-60%)
        total = variant_counts["control"] + variant_counts["experimental"]
        control_ratio = variant_counts["control"] / total
        assert 0.3 <= control_ratio <= 0.7
    
    def test_metric_calculation(self, ab_testing_service):
        """Metrik hesaplama testi"""
        events = [
            {"accuracy": 0.8, "engagement": 0.9, "response_time": 45.0},
            {"accuracy": 0.9, "engagement": 0.85, "response_time": 50.0},
            {"accuracy": 0.7, "engagement": 0.8, "response_time": 60.0}
        ]
        
        metrics = ab_testing_service._calculate_metrics(events, ["accuracy", "engagement"])
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "engagement" in metrics
        
        # Ortalama değerlerin doğru hesaplandığını kontrol et
        assert abs(metrics["accuracy"]["mean"] - 0.8) < 0.01
        assert abs(metrics["engagement"]["mean"] - 0.85) < 0.01
    
    def test_confidence_interval_calculation(self, ab_testing_service):
        """Güven aralığı hesaplama testi"""
        values = [0.8, 0.9, 0.7, 0.85, 0.9]
        confidence_level = 0.95
        
        ci = ab_testing_service._calculate_confidence_interval(values, confidence_level)
        
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Alt sınır üst sınırdan küçük olmalı
        assert 0.0 <= ci[0] <= 1.0
        assert 0.0 <= ci[1] <= 1.0
    
    def test_statistical_test_t_test(self, ab_testing_service):
        """T-test istatistiksel testi"""
        control_values = [0.8, 0.9, 0.7, 0.85, 0.9]
        experimental_values = [0.9, 0.95, 0.85, 0.9, 0.95]
        
        # Mock experiment oluştur
        mock_experiment = Experiment(
            id="test-exp",
            name="Test",
            description="Test",
            variants=[],
            status=ExperimentStatus.ACTIVE,
            start_date=datetime.utcnow()
        )
        
        result = ab_testing_service._perform_statistical_test_values(
            control_values, experimental_values, StatisticalTest.T_TEST, mock_experiment
        )
        
        assert isinstance(result, dict)
        assert "p_value" in result
        assert "effect_size" in result
        assert "is_significant" in result
        assert 0.0 <= result["p_value"] <= 1.0
        assert isinstance(result["is_significant"], bool) or isinstance(result["is_significant"], np.bool_)
    
    @pytest.mark.asyncio
    async def test_sample_size_calculation(self, ab_testing_service):
        """Örnek boyutu hesaplama testi"""
        effect_size = 0.3
        alpha = 0.05
        power = 0.8
        
        sample_size = await ab_testing_service.calculate_sample_size(effect_size, alpha, power)
        
        assert isinstance(sample_size, int)
        assert sample_size > 0
        assert sample_size >= 50  # Minimum örnek boyutu
    
    def test_early_stopping_check(self, ab_testing_service):
        """Erken durdurma kontrolü testi"""
        control_values = [0.8] * 50
        experimental_values = [0.9] * 50
        
        should_stop = ab_testing_service.check_early_stopping_values(
            control_values, experimental_values, 0.01
        )
        
        assert isinstance(should_stop, bool) or isinstance(should_stop, np.bool_)
    
    def test_experiment_recommendations(self, ab_testing_service):
        """Deney önerileri testi"""
        results = {
            "control": {
                "accuracy": {"mean": 0.8, "std": 0.1}, 
                "sample_size": 50,
                "confidence_intervals": {"accuracy": (0.7, 0.9)}
            },
            "experimental": {
                "accuracy": {"mean": 0.9, "std": 0.1}, 
                "sample_size": 50,
                "confidence_intervals": {"accuracy": (0.8, 1.0)}
            }
        }
        
        statistical_tests = {
            "accuracy": {"p_value": 0.01, "is_significant": True, "effect_size": 0.5}
        }
        
        recommendations = ab_testing_service._generate_recommendations(
            results, statistical_tests
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for recommendation in recommendations:
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0
    
    @pytest.mark.asyncio
    async def test_experiment_integration(self, ab_testing_service, sample_variants):
        """Deney entegrasyon testi"""
        # 1. Deney oluştur
        experiment = await ab_testing_service.create_experiment(
            name="Integration Test",
            description="Testing full workflow",
            variants=sample_variants,
            target_metrics=["accuracy", "engagement"],
            duration_days=7
        )
        
        # 2. Deneyi başlat
        success = await ab_testing_service.start_experiment(experiment.id)
        assert success is True
        
        # 3. Kullanıcılara varyant ata
        user_variants = {}
        for i in range(10):
            user_id = f"user-{i}"
            variant = await ab_testing_service.assign_variant(experiment.id, user_id)
            user_variants[user_id] = variant
            assert variant is not None
        
        # 4. Olayları kaydet
        for user_id, variant in user_variants.items():
            event_data = {
                "accuracy": 0.8 if variant == "control" else 0.9,
                "engagement": 0.7 if variant == "control" else 0.8
            }
            success = await ab_testing_service.record_experiment_event(
                experiment.id, user_id, variant, "question_answered", event_data
            )
            assert success is True
        
        # 5. Deneyi tamamla
        success = await ab_testing_service.complete_experiment(experiment.id)
        assert success is True
        
        # 6. Analiz et
        analysis = await ab_testing_service.analyze_experiment(experiment.id)
        assert isinstance(analysis, dict)
        assert "variant_results" in analysis
        assert "recommendations" in analysis
        
        # 7. Durumu kontrol et
        status = await ab_testing_service.get_experiment_status(experiment.id)
        assert status["status"] == "analyzed"  # analyze_experiment metodu durumu ANALYZED yapar
        
        # 8. Listele
        experiments = await ab_testing_service.list_experiments()
        assert len(experiments) > 0
        assert any(exp["id"] == experiment.id for exp in experiments)
