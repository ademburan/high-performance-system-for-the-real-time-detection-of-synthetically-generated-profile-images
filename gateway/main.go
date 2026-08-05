// Package main - GANEye Detection API Gateway (v2).
//

package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	amqp "github.com/rabbitmq/amqp091-go"
)

// --- Prometheus metrikleri --------------------------------------------------

var (
	mRequests = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "ganeye_gateway_requests_total",
		Help: "Gateway'e gelen toplam istek sayısı, HTTP status koduna göre.",
	}, []string{"status"})

	mReqDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "ganeye_gateway_request_duration_seconds",
		Help:    "İstek işleme süresi (multipart parse + publish dahil).",
		Buckets: prometheus.DefBuckets,
	})

	mInFlight = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "ganeye_gateway_in_flight",
		Help: "Şu an işlenmekte olan eşzamanlı istek sayısı.",
	})

	mPublishErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ganeye_gateway_publish_errors_total",
		Help: "RabbitMQ publish hatası/timeout sayısı.",
	})
)

// Task - 'tasks' kuyruğuna gönderilen mesaj şeması.
type Task struct {
	UUID      string `json:"uuid"`
	ImageB64  string `json:"image_b64"`
	Filename  string `json:"filename"`
	Timestamp int64  `json:"timestamp"`
}

type Config struct {
	RabbitURL      string
	HTTPPort       string
	MaxUploadBytes int64
	PublishTimeout time.Duration
	MaxConcurrent  int
}

func loadConfig() Config {
	mb, _ := strconv.Atoi(getenv("MAX_UPLOAD_MB", "10"))
	pt, _ := strconv.Atoi(getenv("PUBLISH_TIMEOUT_MS", "2000"))
	mc, _ := strconv.Atoi(getenv("MAX_CONCURRENT_REQUESTS", "256"))
	return Config{
		RabbitURL:      getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"),
		HTTPPort:       getenv("HTTP_PORT", "8080"),
		MaxUploadBytes: int64(mb) * 1024 * 1024,
		PublishTimeout: time.Duration(pt) * time.Millisecond,
		MaxConcurrent:  mc,
	}
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

// Publisher - tek kanal, serialize publish. v2: done channel ile temiz kapanış.
type Publisher struct {
	conn    *amqp.Connection
	ch      *amqp.Channel
	queue   string
	jobs    chan publishJob
	closing chan struct{}
	done    chan struct{} // DÜZELTME: loop'un bittiğini garanti eder
}

type publishJob struct {
	body []byte
	done chan error
	ctx  context.Context
}

func NewPublisher(url, queue string) (*Publisher, error) {
	var conn *amqp.Connection
	var err error
	for i := 0; i < 30; i++ {
		conn, err = amqp.Dial(url)
		if err == nil {
			break
		}
		log.Printf("[gateway] RabbitMQ bağlantı denemesi %d başarısız: %v", i+1, err)
		time.Sleep(time.Second)
	}
	if err != nil {
		return nil, fmt.Errorf("rabbitmq connect: %w", err)
	}

	ch, err := conn.Channel()
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("open channel: %w", err)
	}
	if _, err = ch.QueueDeclare(queue, true, false, false, false, nil); err != nil {
		ch.Close()
		conn.Close()
		return nil, fmt.Errorf("declare queue: %w", err)
	}

	p := &Publisher{
		conn:    conn,
		ch:      ch,
		queue:   queue,
		jobs:    make(chan publishJob, 1024),
		closing: make(chan struct{}),
		done:    make(chan struct{}),
	}
	go p.loop()
	return p, nil
}

func (p *Publisher) loop() {
	defer close(p.done) // DÜZELTME: kapanışı sinyalle
	for {
		select {
		case <-p.closing:
			return
		case job := <-p.jobs:
			err := p.ch.PublishWithContext(
				job.ctx, "", p.queue, false, false,
				amqp.Publishing{
					ContentType:  "application/json",
					Body:         job.body,
					DeliveryMode: amqp.Persistent,
					Timestamp:    time.Now(),
				},
			)
			job.done <- err
		}
	}
}

func (p *Publisher) Publish(ctx context.Context, body []byte) error {
	done := make(chan error, 1)
	select {
	case p.jobs <- publishJob{body: body, done: done, ctx: ctx}:
	case <-ctx.Done():
		return ctx.Err()
	}
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Close - DÜZELTME: önce loop'u durdur, bitmesini BEKLE, sonra kanalı kapat.
func (p *Publisher) Close() {
	close(p.closing)
	<-p.done // loop kesin bitti, artık ch.Close() güvenli
	if p.ch != nil {
		_ = p.ch.Close()
	}
	if p.conn != nil {
		_ = p.conn.Close()
	}
}

type gateway struct {
	cfg Config
	pub *Publisher
	sem chan struct{}
}

// respond - status'u hem yazar hem Prometheus'a sayar.
func respond(w http.ResponseWriter, status int, body string) {
	mRequests.WithLabelValues(strconv.Itoa(status)).Inc()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write([]byte(body))
}

func (g *gateway) limit(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		select {
		case g.sem <- struct{}{}:
			defer func() { <-g.sem }()
			mInFlight.Inc()
			defer mInFlight.Dec()
			next(w, r)
		default:
			respond(w, http.StatusServiceUnavailable, `{"error":"server_overloaded"}`)
		}
	}
}

func (g *gateway) handleDetect(w http.ResponseWriter, r *http.Request) {
	t0 := time.Now()
	defer func() { mReqDuration.Observe(time.Since(t0).Seconds()) }()

	if r.Method != http.MethodPost {
		respond(w, http.StatusMethodNotAllowed, `{"error":"method_not_allowed"}`)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, g.cfg.MaxUploadBytes)
	if err := r.ParseMultipartForm(g.cfg.MaxUploadBytes); err != nil {
		respond(w, http.StatusBadRequest, `{"error":"invalid_multipart_or_too_large"}`)
		return
	}

	file, header, err := r.FormFile("image")
	if err != nil {
		respond(w, http.StatusBadRequest, `{"error":"image_field_required"}`)
		return
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil || len(data) == 0 {
		respond(w, http.StatusBadRequest, `{"error":"empty_or_unreadable_upload"}`)
		return
	}

	id := uuid.NewString()
	body, err := json.Marshal(Task{
		UUID:      id,
		ImageB64:  base64.StdEncoding.EncodeToString(data),
		Filename:  header.Filename,
		Timestamp: time.Now().UnixMilli(),
	})
	if err != nil {
		respond(w, http.StatusInternalServerError, `{"error":"marshal_failed"}`)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), g.cfg.PublishTimeout)
	defer cancel()

	if err := g.pub.Publish(ctx, body); err != nil {
		log.Printf("[gateway] publish error uuid=%s err=%v", id, err)
		mPublishErrors.Inc()
		respond(w, http.StatusGatewayTimeout, `{"error":"publish_timeout"}`)
		return
	}

	mRequests.WithLabelValues("202").Inc()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"uuid": id, "status": "queued", "bytes": len(data),
	})
}

func (g *gateway) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

func main() {
	cfg := loadConfig()
	log.Printf("[gateway] starting port=%s max_upload=%dMB max_concurrent=%d",
		cfg.HTTPPort, cfg.MaxUploadBytes/(1024*1024), cfg.MaxConcurrent)

	pub, err := NewPublisher(cfg.RabbitURL, "tasks")
	if err != nil {
		log.Fatalf("[gateway] publisher init failed: %v", err)
	}
	defer pub.Close()

	g := &gateway{cfg: cfg, pub: pub, sem: make(chan struct{}, cfg.MaxConcurrent)}

	mux := http.NewServeMux()
	mux.HandleFunc("/detect", g.limit(g.handleDetect))
	mux.HandleFunc("/health", g.handleHealth)
	mux.Handle("/metrics", promhttp.Handler()) // Prometheus scrape endpoint'i

	srv := &http.Server{
		Addr:         ":" + cfg.HTTPPort,
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	idleConnsClosed := make(chan struct{})
	go func() {
		sigint := make(chan os.Signal, 1)
		signal.Notify(sigint, syscall.SIGINT, syscall.SIGTERM)
		<-sigint
		log.Println("[gateway] shutdown signal received")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
		close(idleConnsClosed)
	}()

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("[gateway] server error: %v", err)
	}
	<-idleConnsClosed
	log.Println("[gateway] stopped")
}
