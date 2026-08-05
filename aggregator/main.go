// Package main - GANEye Detection Aggregator .
//
//
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	amqp "github.com/rabbitmq/amqp091-go"
)

// --- Prometheus metrikleri --------------------------------------------------

var (
	mPersisted = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ganeye_aggregator_persisted_total",
		Help: "PostgreSQL'e başarıyla yazılan sonuç sayısı.",
	})
	mInsertErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ganeye_aggregator_insert_errors_total",
		Help: "Başarısız insert denemesi sayısı (nack+requeue edilir).",
	})
	mConsumerRestarts = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ganeye_aggregator_consumer_restarts_total",
		Help: "Consumer loop'unun yeniden başlatılma sayısı.",
	})
)

// Result - worker'dan gelen sonuç şeması.
type Result struct {
	UUID         string   `json:"uuid"`
	Score        float64  `json:"score"`
	Label        string   `json:"label"`
	Error        *string  `json:"error"`
	Detail       *string  `json:"detail,omitempty"`
	FaceCount    *int     `json:"face_count,omitempty"`
	Filename     string   `json:"filename,omitempty"`
	ProcessingMS *int     `json:"processing_ms,omitempty"`
	ImageW       *int     `json:"image_w,omitempty"`
	ImageH       *int     `json:"image_h,omitempty"`
	Threshold    *float64 `json:"threshold,omitempty"`
}

type Config struct {
	RabbitURL   string
	ResultQueue string
	DatabaseURL string
	HTTPPort    string
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func loadConfig() Config {
	return Config{
		RabbitURL:   getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"),
		ResultQueue: getenv("RESULT_QUEUE", "results"),
		DatabaseURL: getenv("DATABASE_URL", "postgres://ganeye:ganeye@localhost:5432/ganeye"),
		HTTPPort:    getenv("HTTP_PORT", "8081"),
	}
}

// initDB - PostgreSQL bağlantı havuzunu açar, şemayı garanti eder.
// Postgres yavaş başlayabilir; 30 sn retry.
func initDB(ctx context.Context, url string) (*pgxpool.Pool, error) {
	var pool *pgxpool.Pool
	var err error
	for i := 0; i < 30; i++ {
		pool, err = pgxpool.New(ctx, url)
		if err == nil {
			if pingErr := pool.Ping(ctx); pingErr == nil {
				break
			} else {
				err = pingErr
				pool.Close()
				pool = nil
			}
		}
		log.Printf("[aggregator] postgres bağlantı denemesi %d: %v", i+1, err)
		time.Sleep(time.Second)
	}
	if pool == nil {
		return nil, fmt.Errorf("postgres bağlanamadı: %w", err)
	}

	schema := `
	CREATE TABLE IF NOT EXISTS results (
		uuid          TEXT PRIMARY KEY,
		score         DOUBLE PRECISION NOT NULL,
		label         TEXT NOT NULL,
		error         TEXT,
		detail        TEXT,
		face_count    INTEGER,
		filename      TEXT,
		processing_ms INTEGER,
		image_w       INTEGER,
		image_h       INTEGER,
		threshold     DOUBLE PRECISION,
		received_at   BIGINT NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_results_label    ON results(label);
	CREATE INDEX IF NOT EXISTS idx_results_received ON results(received_at);`
	if _, err := pool.Exec(ctx, schema); err != nil {
		pool.Close()
		return nil, fmt.Errorf("şema oluşturulamadı: %w", err)
	}
	return pool, nil
}

func dialRabbit(url string) (*amqp.Connection, error) {
	var err error
	for i := 0; i < 30; i++ {
		conn, dErr := amqp.Dial(url)
		if dErr == nil {
			return conn, nil
		}
		err = dErr
		log.Printf("[aggregator] RabbitMQ dial %d: %v", i+1, dErr)
		time.Sleep(time.Second)
	}
	return nil, err
}

// insertResult - UPSERT: aynı UUID tekrar gelirse (at-least-once delivery)
// son sonuç öncekini ezer. PostgreSQL placeholder'ları $1..$12.
func insertResult(ctx context.Context, pool *pgxpool.Pool, r Result) error {
	const q = `
	INSERT INTO results (
		uuid, score, label, error, detail, face_count, filename,
		processing_ms, image_w, image_h, threshold, received_at
	) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
	ON CONFLICT (uuid) DO UPDATE SET
		score=EXCLUDED.score, label=EXCLUDED.label, error=EXCLUDED.error,
		detail=EXCLUDED.detail, face_count=EXCLUDED.face_count,
		filename=EXCLUDED.filename, processing_ms=EXCLUDED.processing_ms,
		image_w=EXCLUDED.image_w, image_h=EXCLUDED.image_h,
		threshold=EXCLUDED.threshold, received_at=EXCLUDED.received_at;`

	_, err := pool.Exec(ctx, q,
		r.UUID, r.Score, r.Label, r.Error, r.Detail,
		r.FaceCount, r.Filename, r.ProcessingMS,
		r.ImageW, r.ImageH, r.Threshold,
		time.Now().UnixMilli(),
	)
	return err
}

// consumeOnce - tek bir consume oturumu. Kanal/bağlantı hatasında error
// döner; üst katman yeniden başlatır.
func consumeOnce(ctx context.Context, conn *amqp.Connection, pool *pgxpool.Pool, queue string, processed *atomic.Int64) error {
	ch, err := conn.Channel()
	if err != nil {
		return fmt.Errorf("open channel: %w", err)
	}
	defer ch.Close()

	if _, err := ch.QueueDeclare(queue, true, false, false, false, nil); err != nil {
		return fmt.Errorf("declare: %w", err)
	}
	if err := ch.Qos(32, 0, false); err != nil {
		return fmt.Errorf("qos: %w", err)
	}

	msgs, err := ch.Consume(queue, "aggregator", false, false, false, false, nil)
	if err != nil {
		return fmt.Errorf("consume: %w", err)
	}

	log.Printf("[aggregator] consuming queue=%s", queue)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case d, ok := <-msgs:
			if !ok {
				return fmt.Errorf("consumer channel closed")
			}
			var r Result
			if err := json.Unmarshal(d.Body, &r); err != nil {
				log.Printf("[aggregator] invalid JSON, dropping: %v", err)
				_ = d.Ack(false) // bozuk mesaj kuyruğu tıkamasın
				continue
			}
			if err := insertResult(ctx, pool, r); err != nil {
				log.Printf("[aggregator] insert failed uuid=%s err=%v", r.UUID, err)
				mInsertErrors.Inc()
				_ = d.Nack(false, true) // requeue
				continue
			}
			_ = d.Ack(false)
			processed.Add(1)
			mPersisted.Inc()
		}
	}
}

// runConsumer - DÜZELTME: consumer ölürse 3 sn bekleyip yeniden başlatır.
// Gerekirse RabbitMQ bağlantısını da yeniden kurar. v1'deki "sessiz ölüm"
// problemi bu döngüyle çözülür.
func runConsumer(ctx context.Context, cfg Config, pool *pgxpool.Pool, processed *atomic.Int64) {
	for {
		if ctx.Err() != nil {
			return
		}
		conn, err := dialRabbit(cfg.RabbitURL)
		if err != nil {
			log.Printf("[aggregator] rabbit dial failed, retrying: %v", err)
			time.Sleep(3 * time.Second)
			continue
		}
		err = consumeOnce(ctx, conn, pool, cfg.ResultQueue, processed)
		_ = conn.Close()
		if ctx.Err() != nil {
			return
		}
		mConsumerRestarts.Inc()
		log.Printf("[aggregator] consumer exited (%v), restarting in 3s", err)
		time.Sleep(3 * time.Second)
	}
}

// --- HTTP API ----------------------------------------------------------------

type apiServer struct {
	pool      *pgxpool.Pool
	processed *atomic.Int64
}

func (s *apiServer) handleStats(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	var total, bots, real_, unknown int
	_ = s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM results`).Scan(&total)
	_ = s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM results WHERE label='bot'`).Scan(&bots)
	_ = s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM results WHERE label='real'`).Scan(&real_)
	_ = s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM results WHERE label='unknown'`).Scan(&unknown)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"total":             total,
		"bot":               bots,
		"real":              real_,
		"unknown":           unknown,
		"processed_session": s.processed.Load(),
	})
}

func (s *apiServer) handleByUUID(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/results/")
	if id == "" {
		http.Error(w, `{"error":"uuid_required"}`, http.StatusBadRequest)
		return
	}
	row := s.pool.QueryRow(r.Context(),
		`SELECT uuid, score, label, COALESCE(error,''), COALESCE(filename,'')
		 FROM results WHERE uuid=$1`, id)
	var uuid, label, errStr, fn string
	var score float64
	if err := row.Scan(&uuid, &score, &label, &errStr, &fn); err != nil {
		http.Error(w, `{"error":"not_found"}`, http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"uuid": uuid, "score": score, "label": label, "error": errStr, "filename": fn,
	})
}

func (s *apiServer) handleList(w http.ResponseWriter, r *http.Request) {
	limit := 100
	if v := r.URL.Query().Get("limit"); v != "" {
		fmt.Sscanf(v, "%d", &limit)
		if limit <= 0 || limit > 10000 {
			limit = 100
		}
	}
	rows, err := s.pool.Query(r.Context(),
		`SELECT uuid, score, label, COALESCE(error,''), COALESCE(filename,''), received_at
		 FROM results ORDER BY received_at DESC LIMIT $1`, limit)
	if err != nil {
		http.Error(w, `{"error":"db_error"}`, http.StatusInternalServerError)
		return
	}
	defer rows.Close()
	out := make([]map[string]any, 0, limit)
	for rows.Next() {
		var uuid, label, errStr, fn string
		var score float64
		var ts int64
		if err := rows.Scan(&uuid, &score, &label, &errStr, &fn, &ts); err != nil {
			continue
		}
		out = append(out, map[string]any{
			"uuid": uuid, "score": score, "label": label,
			"error": errStr, "filename": fn, "received_at": ts,
		})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

func main() {
	cfg := loadConfig()
	log.Printf("[aggregator] starting queue=%s http=:%s", cfg.ResultQueue, cfg.HTTPPort)

	rootCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pool, err := initDB(rootCtx, cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("[aggregator] db init: %v", err)
	}
	defer pool.Close()

	var processed atomic.Int64

	// Kendini yeniden başlatan consumer döngüsü.
	go runConsumer(rootCtx, cfg, pool, &processed)

	api := &apiServer{pool: pool, processed: &processed}
	mux := http.NewServeMux()
	mux.HandleFunc("/stats", api.handleStats)
	mux.HandleFunc("/results/", api.handleByUUID)
	mux.HandleFunc("/results", api.handleList)
	mux.Handle("/metrics", promhttp.Handler()) // Prometheus scrape endpoint'i
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"status":"ok"}`))
	})

	srv := &http.Server{
		Addr:         ":" + cfg.HTTPPort,
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 30 * time.Second,
	}

	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		log.Println("[aggregator] shutting down")
		cancel()
		sc, c := context.WithTimeout(context.Background(), 10*time.Second)
		defer c()
		_ = srv.Shutdown(sc)
	}()

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("[aggregator] http: %v", err)
	}
	log.Println("[aggregator] stopped")
}
