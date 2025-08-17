from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import json
import time
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics

from app.core.cache import cache_service
from app.core.database import database

logger = logging.getLogger(__name__)


class PerformanceMonitoringService:
    """Comprehensive performance monitoring and metrics tracking service"""
    
    def __init__(self):
        self.cache_ttl = 3600  # 1 hour
        self.metrics_retention_days = 30
        self.alert_thresholds = {
            "latency_p95": 2.0,  # seconds
            "error_rate": 0.05,  # 5%
            "cost_per_request": 0.01,  # dollars
            "quality_score": 0.7
        }
        
        # Metrics storage
        self.latency_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.error_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.cost_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.quality_metrics = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance counters
        self.request_counters = defaultdict(int)
        self.error_counters = defaultdict(int)
        self.cost_accumulators = defaultdict(float)
        
        # Alert history
        self.alert_history = deque(maxlen=100)
    
    async def track_request(
        self,
        operation_type: str,
        start_time: float,
        end_time: float,
        success: bool,
        error_message: Optional[str] = None,
        cost: Optional[float] = None,
        quality_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track a single request/operation"""
        
        try:
            # Calculate latency
            latency = end_time - start_time
            
            # Record metrics
            self.latency_metrics[operation_type].append(latency)
            self.error_metrics[operation_type].append(0 if success else 1)
            
            if cost is not None:
                self.cost_metrics[operation_type].append(cost)
                self.cost_accumulators[operation_type] += cost
            
            if quality_score is not None:
                self.quality_metrics[operation_type].append(quality_score)
            
            # Update counters
            self.request_counters[operation_type] += 1
            if not success:
                self.error_counters[operation_type] += 1
            
            # Check for alerts
            alerts = await self._check_alerts(operation_type, latency, success, cost, quality_score)
            
            # Store detailed metrics
            metric_record = {
                "operation_type": operation_type,
                "timestamp": datetime.now().isoformat(),
                "latency": latency,
                "success": success,
                "error_message": error_message,
                "cost": cost,
                "quality_score": quality_score,
                "metadata": metadata or {},
                "alerts": alerts
            }
            
            # Cache recent metrics
            await self._cache_metrics(operation_type, metric_record)
            
            return {
                "tracked": True,
                "latency": latency,
                "alerts": alerts,
                "metric_id": f"{operation_type}_{int(start_time * 1000)}"
            }
            
        except Exception as e:
            logger.error(f"Error tracking request: {e}")
            return {
                "tracked": False,
                "error": str(e)
            }
    
    async def get_performance_metrics(
        self,
        operation_type: Optional[str] = None,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        try:
            # Calculate time range
            end_time = datetime.now()
            if time_range == "1h":
                start_time = end_time - timedelta(hours=1)
            elif time_range == "24h":
                start_time = end_time - timedelta(days=1)
            elif time_range == "7d":
                start_time = end_time - timedelta(days=7)
            elif time_range == "30d":
                start_time = end_time - timedelta(days=30)
            else:
                start_time = end_time - timedelta(hours=24)
            
            # Get metrics from database
            metrics = await self._get_metrics_from_db(operation_type, start_time, end_time)
            
            # Calculate aggregated metrics
            aggregated_metrics = self._calculate_aggregated_metrics(metrics, operation_type)
            
            # Get real-time metrics
            realtime_metrics = self._get_realtime_metrics(operation_type)
            
            # Get alert status
            alert_status = await self._get_alert_status(operation_type)
            
            return {
                "time_range": time_range,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "aggregated_metrics": aggregated_metrics,
                "realtime_metrics": realtime_metrics,
                "alert_status": alert_status,
                "operation_types": list(self.request_counters.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {
                "error": str(e),
                "time_range": time_range,
                "aggregated_metrics": {},
                "realtime_metrics": {},
                "alert_status": {}
            }
    
    async def get_latency_analysis(
        self,
        operation_type: str,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Get detailed latency analysis"""
        
        try:
            # Get latency data
            latency_data = list(self.latency_metrics[operation_type])
            
            if not latency_data:
                return {
                    "operation_type": operation_type,
                    "message": "No latency data available",
                    "percentiles": {},
                    "trends": {}
                }
            
            # Calculate percentiles
            percentiles = {
                "p50": statistics.quantiles(latency_data, n=2)[0],
                "p90": statistics.quantiles(latency_data, n=10)[8],
                "p95": statistics.quantiles(latency_data, n=20)[18],
                "p99": statistics.quantiles(latency_data, n=100)[98]
            }
            
            # Calculate trends
            recent_data = latency_data[-100:] if len(latency_data) > 100 else latency_data
            older_data = latency_data[:-100] if len(latency_data) > 100 else []
            
            if older_data:
                recent_avg = statistics.mean(recent_data)
                older_avg = statistics.mean(older_data)
                trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
            else:
                trend = 0
            
            return {
                "operation_type": operation_type,
                "total_requests": len(latency_data),
                "percentiles": percentiles,
                "trends": {
                    "recent_avg": statistics.mean(recent_data),
                    "overall_avg": statistics.mean(latency_data),
                    "trend_percentage": trend * 100,
                    "trend_direction": "improving" if trend < 0 else "degrading" if trend > 0 else "stable"
                },
                "distribution": {
                    "min": min(latency_data),
                    "max": max(latency_data),
                    "std_dev": statistics.stdev(latency_data) if len(latency_data) > 1 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in latency analysis: {e}")
            return {
                "operation_type": operation_type,
                "error": str(e)
            }
    
    async def get_cost_analysis(
        self,
        operation_type: Optional[str] = None,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Get detailed cost analysis"""
        
        try:
            # Calculate time range
            end_time = datetime.now()
            if time_range == "24h":
                start_time = end_time - timedelta(days=1)
            elif time_range == "7d":
                start_time = end_time - timedelta(days=7)
            elif time_range == "30d":
                start_time = end_time - timedelta(days=30)
            else:
                start_time = end_time - timedelta(hours=24)
            
            # Get cost data from database
            cost_data = await self._get_cost_data_from_db(operation_type, start_time, end_time)
            
            if not cost_data:
                return {
                    "time_range": time_range,
                    "message": "No cost data available",
                    "total_cost": 0.0,
                    "cost_per_request": 0.0,
                    "cost_breakdown": {}
                }
            
            # Calculate cost metrics
            total_cost = sum(record["cost"] for record in cost_data)
            total_requests = len(cost_data)
            cost_per_request = total_cost / total_requests if total_requests > 0 else 0
            
            # Cost breakdown by operation type
            cost_breakdown = defaultdict(float)
            for record in cost_data:
                op_type = record["operation_type"]
                cost_breakdown[op_type] += record["cost"]
            
            # Cost trends
            daily_costs = defaultdict(float)
            for record in cost_data:
                date = record["timestamp"][:10]  # Extract date part
                daily_costs[date] += record["cost"]
            
            return {
                "time_range": time_range,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_cost": total_cost,
                "total_requests": total_requests,
                "cost_per_request": cost_per_request,
                "cost_breakdown": dict(cost_breakdown),
                "daily_costs": dict(daily_costs),
                "cost_efficiency": {
                    "avg_cost_per_request": cost_per_request,
                    "cost_trend": self._calculate_cost_trend(daily_costs)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in cost analysis: {e}")
            return {
                "time_range": time_range,
                "error": str(e),
                "total_cost": 0.0,
                "cost_per_request": 0.0
            }
    
    async def get_quality_metrics(
        self,
        operation_type: Optional[str] = None,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Get quality metrics analysis"""
        
        try:
            # Get quality data
            quality_data = []
            if operation_type:
                quality_data = list(self.quality_metrics[operation_type])
            else:
                for op_type in self.quality_metrics:
                    quality_data.extend(self.quality_metrics[op_type])
            
            if not quality_data:
                return {
                    "time_range": time_range,
                    "message": "No quality data available",
                    "overall_quality": 0.0,
                    "quality_breakdown": {}
                }
            
            # Calculate quality metrics
            overall_quality = statistics.mean(quality_data)
            
            # Quality breakdown by operation type
            quality_breakdown = {}
            for op_type in self.quality_metrics:
                if self.quality_metrics[op_type]:
                    quality_breakdown[op_type] = {
                        "avg_quality": statistics.mean(self.quality_metrics[op_type]),
                        "min_quality": min(self.quality_metrics[op_type]),
                        "max_quality": max(self.quality_metrics[op_type]),
                        "sample_count": len(self.quality_metrics[op_type])
                    }
            
            # Quality trends
            recent_quality = quality_data[-100:] if len(quality_data) > 100 else quality_data
            older_quality = quality_data[:-100] if len(quality_data) > 100 else []
            
            if older_quality:
                recent_avg = statistics.mean(recent_quality)
                older_avg = statistics.mean(older_quality)
                quality_trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
            else:
                quality_trend = 0
            
            return {
                "time_range": time_range,
                "overall_quality": overall_quality,
                "quality_breakdown": quality_breakdown,
                "quality_trends": {
                    "recent_avg": statistics.mean(recent_quality),
                    "trend_percentage": quality_trend * 100,
                    "trend_direction": "improving" if quality_trend > 0 else "degrading" if quality_trend < 0 else "stable"
                },
                "quality_distribution": {
                    "excellent": len([q for q in quality_data if q >= 0.9]),
                    "good": len([q for q in quality_data if 0.7 <= q < 0.9]),
                    "acceptable": len([q for q in quality_data if 0.5 <= q < 0.7]),
                    "poor": len([q for q in quality_data if q < 0.5])
                }
            }
            
        except Exception as e:
            logger.error(f"Error in quality metrics: {e}")
            return {
                "time_range": time_range,
                "error": str(e),
                "overall_quality": 0.0
            }
    
    async def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current alerts"""
        
        try:
            alerts = []
            
            # Check latency alerts
            for op_type in self.latency_metrics:
                if self.latency_metrics[op_type]:
                    recent_latencies = list(self.latency_metrics[op_type])[-100:]
                    p95_latency = statistics.quantiles(recent_latencies, n=20)[18] if len(recent_latencies) >= 20 else statistics.mean(recent_latencies)
                    
                    if p95_latency > self.alert_thresholds["latency_p95"]:
                        alerts.append({
                            "type": "latency",
                            "operation_type": op_type,
                            "severity": "high" if p95_latency > self.alert_thresholds["latency_p95"] * 2 else "medium",
                            "message": f"P95 latency for {op_type} is {p95_latency:.2f}s (threshold: {self.alert_thresholds['latency_p95']}s)",
                            "value": p95_latency,
                            "threshold": self.alert_thresholds["latency_p95"],
                            "timestamp": datetime.now().isoformat()
                        })
            
            # Check error rate alerts
            for op_type in self.error_metrics:
                if self.error_metrics[op_type]:
                    recent_errors = list(self.error_metrics[op_type])[-100:]
                    error_rate = sum(recent_errors) / len(recent_errors)
                    
                    if error_rate > self.alert_thresholds["error_rate"]:
                        alerts.append({
                            "type": "error_rate",
                            "operation_type": op_type,
                            "severity": "high" if error_rate > self.alert_thresholds["error_rate"] * 2 else "medium",
                            "message": f"Error rate for {op_type} is {error_rate:.2%} (threshold: {self.alert_thresholds['error_rate']:.2%})",
                            "value": error_rate,
                            "threshold": self.alert_thresholds["error_rate"],
                            "timestamp": datetime.now().isoformat()
                        })
            
            # Check cost alerts
            for op_type in self.cost_metrics:
                if self.cost_metrics[op_type]:
                    recent_costs = list(self.cost_metrics[op_type])[-100:]
                    avg_cost = statistics.mean(recent_costs)
                    
                    if avg_cost > self.alert_thresholds["cost_per_request"]:
                        alerts.append({
                            "type": "cost",
                            "operation_type": op_type,
                            "severity": "high" if avg_cost > self.alert_thresholds["cost_per_request"] * 2 else "medium",
                            "message": f"Average cost for {op_type} is ${avg_cost:.4f} (threshold: ${self.alert_thresholds['cost_per_request']:.4f})",
                            "value": avg_cost,
                            "threshold": self.alert_thresholds["cost_per_request"],
                            "timestamp": datetime.now().isoformat()
                        })
            
            # Check quality alerts
            for op_type in self.quality_metrics:
                if self.quality_metrics[op_type]:
                    recent_quality = list(self.quality_metrics[op_type])[-100:]
                    avg_quality = statistics.mean(recent_quality)
                    
                    if avg_quality < self.alert_thresholds["quality_score"]:
                        alerts.append({
                            "type": "quality",
                            "operation_type": op_type,
                            "severity": "high" if avg_quality < self.alert_thresholds["quality_score"] * 0.7 else "medium",
                            "message": f"Quality score for {op_type} is {avg_quality:.2f} (threshold: {self.alert_thresholds['quality_score']:.2f})",
                            "value": avg_quality,
                            "threshold": self.alert_thresholds["quality_score"],
                            "timestamp": datetime.now().isoformat()
                        })
            
            # Filter by severity if specified
            if severity:
                alerts = [alert for alert in alerts if alert["severity"] == severity]
            
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    async def _check_alerts(
        self,
        operation_type: str,
        latency: float,
        success: bool,
        cost: Optional[float],
        quality_score: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Check for alerts based on current metrics"""
        
        alerts = []
        
        # Latency alert
        if latency > self.alert_thresholds["latency_p95"]:
            alerts.append({
                "type": "latency",
                "severity": "high" if latency > self.alert_thresholds["latency_p95"] * 2 else "medium",
                "message": f"High latency detected: {latency:.2f}s"
            })
        
        # Error alert
        if not success:
            alerts.append({
                "type": "error",
                "severity": "high",
                "message": "Operation failed"
            })
        
        # Cost alert
        if cost and cost > self.alert_thresholds["cost_per_request"]:
            alerts.append({
                "type": "cost",
                "severity": "medium",
                "message": f"High cost detected: ${cost:.4f}"
            })
        
        # Quality alert
        if quality_score and quality_score < self.alert_thresholds["quality_score"]:
            alerts.append({
                "type": "quality",
                "severity": "medium",
                "message": f"Low quality detected: {quality_score:.2f}"
            })
        
        return alerts
    
    async def _cache_metrics(self, operation_type: str, metric_record: Dict[str, Any]) -> None:
        """Cache metrics for quick access"""
        
        cache_key = f"metrics:{operation_type}:{int(time.time())}"
        await cache_service.set(cache_key, metric_record, self.cache_ttl)
    
    async def _get_metrics_from_db(
        self,
        operation_type: Optional[str],
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get metrics from database"""
        
        try:
            # This would query the actual metrics table
            # For now, return empty list as placeholder
            return []
        except Exception as e:
            logger.error(f"Error getting metrics from DB: {e}")
            return []
    
    async def _get_cost_data_from_db(
        self,
        operation_type: Optional[str],
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get cost data from database"""
        
        try:
            # This would query the actual cost table
            # For now, return empty list as placeholder
            return []
        except Exception as e:
            logger.error(f"Error getting cost data from DB: {e}")
            return []
    
    def _calculate_aggregated_metrics(
        self,
        metrics: List[Dict[str, Any]],
        operation_type: Optional[str]
    ) -> Dict[str, Any]:
        """Calculate aggregated metrics from raw data"""
        
        if not metrics:
            return {}
        
        # Filter by operation type if specified
        if operation_type:
            metrics = [m for m in metrics if m.get("operation_type") == operation_type]
        
        if not metrics:
            return {}
        
        # Calculate basic metrics
        latencies = [m.get("latency", 0) for m in metrics]
        errors = [1 for m in metrics if not m.get("success", True)]
        costs = [m.get("cost", 0) for m in metrics if m.get("cost") is not None]
        qualities = [m.get("quality_score", 0) for m in metrics if m.get("quality_score") is not None]
        
        return {
            "total_requests": len(metrics),
            "successful_requests": len(metrics) - len(errors),
            "error_rate": len(errors) / len(metrics) if metrics else 0,
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "p95_latency": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else statistics.mean(latencies) if latencies else 0,
            "total_cost": sum(costs) if costs else 0,
            "avg_cost": statistics.mean(costs) if costs else 0,
            "avg_quality": statistics.mean(qualities) if qualities else 0
        }
    
    def _get_realtime_metrics(self, operation_type: Optional[str]) -> Dict[str, Any]:
        """Get real-time metrics from memory"""
        
        metrics = {}
        
        if operation_type:
            op_types = [operation_type]
        else:
            op_types = list(self.request_counters.keys())
        
        for op_type in op_types:
            metrics[op_type] = {
                "total_requests": self.request_counters[op_type],
                "total_errors": self.error_counters[op_type],
                "total_cost": self.cost_accumulators[op_type],
                "recent_latencies": list(self.latency_metrics[op_type])[-10:],
                "recent_errors": list(self.error_metrics[op_type])[-10:],
                "recent_costs": list(self.cost_metrics[op_type])[-10:],
                "recent_qualities": list(self.quality_metrics[op_type])[-10:]
            }
        
        return metrics
    
    async def _get_alert_status(self, operation_type: Optional[str]) -> Dict[str, Any]:
        """Get current alert status"""
        
        alerts = await self.get_alerts()
        
        if operation_type:
            alerts = [alert for alert in alerts if alert.get("operation_type") == operation_type]
        
        return {
            "total_alerts": len(alerts),
            "high_severity": len([a for a in alerts if a["severity"] == "high"]),
            "medium_severity": len([a for a in alerts if a["severity"] == "medium"]),
            "alert_types": list(set(a["type"] for a in alerts))
        }
    
    def _calculate_cost_trend(self, daily_costs: Dict[str, float]) -> Dict[str, Any]:
        """Calculate cost trend from daily data"""
        
        if len(daily_costs) < 2:
            return {"trend": "insufficient_data", "percentage": 0}
        
        # Sort by date
        sorted_dates = sorted(daily_costs.keys())
        recent_cost = daily_costs[sorted_dates[-1]]
        older_cost = daily_costs[sorted_dates[-2]]
        
        if older_cost == 0:
            return {"trend": "no_previous_data", "percentage": 0}
        
        trend_percentage = ((recent_cost - older_cost) / older_cost) * 100
        
        return {
            "trend": "increasing" if trend_percentage > 0 else "decreasing" if trend_percentage < 0 else "stable",
            "percentage": trend_percentage
        }
    
    async def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance monitoring statistics"""
        
        return {
            "alert_thresholds": self.alert_thresholds,
            "metrics_retention_days": self.metrics_retention_days,
            "cache_ttl": self.cache_ttl,
            "tracked_operations": list(self.request_counters.keys()),
            "total_requests": sum(self.request_counters.values()),
            "total_errors": sum(self.error_counters.values()),
            "total_cost": sum(self.cost_accumulators.values())
        }


# Global instance
performance_monitoring_service = PerformanceMonitoringService()
