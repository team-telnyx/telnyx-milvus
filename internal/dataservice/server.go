package dataservice

import (
	"context"
	"fmt"
	"log"
	"sync"

	"github.com/zilliztech/milvus-distributed/internal/msgstream"
	"github.com/zilliztech/milvus-distributed/internal/msgstream/pulsarms"

	"github.com/zilliztech/milvus-distributed/internal/distributed/masterservice"

	"github.com/zilliztech/milvus-distributed/internal/proto/milvuspb"

	"github.com/zilliztech/milvus-distributed/internal/timesync"

	etcdkv "github.com/zilliztech/milvus-distributed/internal/kv/etcd"
	"go.etcd.io/etcd/clientv3"

	"github.com/zilliztech/milvus-distributed/internal/proto/commonpb"
	"github.com/zilliztech/milvus-distributed/internal/proto/datapb"
	"github.com/zilliztech/milvus-distributed/internal/proto/internalpb2"
	"github.com/zilliztech/milvus-distributed/internal/util/typeutil"
)

const role = "dataservice"

type DataService interface {
	typeutil.Service
	RegisterNode(req *datapb.RegisterNodeRequest) (*datapb.RegisterNodeResponse, error)
	Flush(req *datapb.FlushRequest) (*commonpb.Status, error)

	AssignSegmentID(req *datapb.AssignSegIDRequest) (*datapb.AssignSegIDResponse, error)
	ShowSegments(req *datapb.ShowSegmentRequest) (*datapb.ShowSegmentResponse, error)
	GetSegmentStates(req *datapb.SegmentStatesRequest) (*datapb.SegmentStatesResponse, error)
	GetInsertBinlogPaths(req *datapb.InsertBinlogPathRequest) (*datapb.InsertBinlogPathsResponse, error)

	GetInsertChannels(req *datapb.InsertChannelRequest) (*internalpb2.StringList, error)
	GetCollectionStatistics(req *datapb.CollectionStatsRequest) (*datapb.CollectionStatsResponse, error)
	GetPartitionStatistics(req *datapb.PartitionStatsRequest) (*datapb.PartitionStatsResponse, error)
	GetComponentStates() (*internalpb2.ComponentStates, error)
	GetTimeTickChannel() (*milvuspb.StringResponse, error)
	GetStatisticsChannel() (*milvuspb.StringResponse, error)
}

type (
	UniqueID  = typeutil.UniqueID
	Timestamp = typeutil.Timestamp
	Server    struct {
		ctx              context.Context
		serverLoopCtx    context.Context
		serverLoopCancel context.CancelFunc
		serverLoopWg     sync.WaitGroup
		state            internalpb2.StateCode
		client           *etcdkv.EtcdKV
		meta             *meta
		segAllocator     segmentAllocator
		statsHandler     *statsHandler
		insertChannelMgr *insertChannelManager
		allocator        allocator
		cluster          *dataNodeCluster
		msgProducer      *timesync.MsgProducer
		registerFinishCh chan struct{}
		masterClient     *masterservice.GrpcClient
		ttMsgStream      msgstream.MsgStream
		ddChannelName    string
	}
)

func CreateServer(ctx context.Context, client *masterservice.GrpcClient) (*Server, error) {
	ch := make(chan struct{})
	return &Server{
		ctx:              ctx,
		state:            internalpb2.StateCode_INITIALIZING,
		insertChannelMgr: newInsertChannelManager(),
		registerFinishCh: ch,
		cluster:          newDataNodeCluster(ch),
		masterClient:     client,
	}, nil
}

func (s *Server) Init() error {
	Params.Init()
	return nil
}

func (s *Server) Start() error {
	s.allocator = newAllocatorImpl(s.masterClient)
	if err := s.initMeta(); err != nil {
		return err
	}
	s.statsHandler = newStatsHandler(s.meta)
	segAllocator, err := newSegmentAllocator(s.meta, s.allocator)
	if err != nil {
		return err
	}
	s.segAllocator = segAllocator
	s.waitDataNodeRegister()

	if err = s.loadMetaFromMaster(); err != nil {
		return err
	}
	if err = s.initMsgProducer(); err != nil {
		return err
	}

	s.startServerLoop()
	s.state = internalpb2.StateCode_HEALTHY
	log.Println("start success")
	return nil
}

func (s *Server) initMeta() error {
	etcdClient, err := clientv3.New(clientv3.Config{Endpoints: []string{Params.EtcdAddress}})
	if err != nil {
		return err
	}
	etcdKV := etcdkv.NewEtcdKV(etcdClient, Params.MetaRootPath)
	s.client = etcdKV
	s.meta, err = newMeta(etcdKV, s.allocator)
	if err != nil {
		return err
	}
	return nil
}

func (s *Server) waitDataNodeRegister() {
	log.Println("waiting data node to register")
	<-s.registerFinishCh
	log.Println("all data nodes register")
}

func (s *Server) initMsgProducer() error {
	s.ttMsgStream = pulsarms.NewPulsarTtMsgStream(s.ctx, 1024)
	s.ttMsgStream.Start()
	timeTickBarrier := timesync.NewHardTimeTickBarrier(s.ttMsgStream, s.cluster.GetNodeIDs())
	dataNodeTTWatcher := newDataNodeTimeTickWatcher(s.meta, s.segAllocator, s.cluster)
	producer, err := timesync.NewTimeSyncMsgProducer(timeTickBarrier, dataNodeTTWatcher)
	if err != nil {
		return err
	}
	s.msgProducer = producer
	s.msgProducer.Start(s.ctx)
	return nil
}

func (s *Server) startServerLoop() {
	s.serverLoopCtx, s.serverLoopCancel = context.WithCancel(s.ctx)
	s.serverLoopWg.Add(1)
	go s.startStatsChannel(s.serverLoopCtx)
}

func (s *Server) startStatsChannel(ctx context.Context) {
	defer s.serverLoopWg.Done()
	statsStream := pulsarms.NewPulsarMsgStream(ctx, 1024)
	statsStream.Start()
	defer statsStream.Close()
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		msgPack := statsStream.Consume()
		for _, msg := range msgPack.Msgs {
			statistics := msg.(*msgstream.SegmentStatisticsMsg)
			for _, stat := range statistics.SegStats {
				if err := s.statsHandler.HandleSegmentStat(stat); err != nil {
					log.Println(err.Error())
					continue
				}
			}
		}
	}
}

func (s *Server) loadMetaFromMaster() error {
	log.Println("loading collection meta from master")
	collections, err := s.masterClient.ShowCollections(&milvuspb.ShowCollectionRequest{
		Base: &commonpb.MsgBase{
			MsgType:   commonpb.MsgType_kShowCollections,
			MsgID:     -1, // todo add msg id
			Timestamp: 0,  // todo
			SourceID:  -1, // todo
		},
		DbName: "",
	})
	if err != nil {
		return err
	}
	for _, collectionName := range collections.CollectionNames {
		collection, err := s.masterClient.DescribeCollection(&milvuspb.DescribeCollectionRequest{
			Base: &commonpb.MsgBase{
				MsgType:   commonpb.MsgType_kDescribeCollection,
				MsgID:     -1, // todo
				Timestamp: 0,  // todo
				SourceID:  -1, // todo
			},
			DbName:         "",
			CollectionName: collectionName,
		})
		if err != nil {
			log.Println(err.Error())
			continue
		}
		partitions, err := s.masterClient.ShowPartitions(&milvuspb.ShowPartitionRequest{
			Base: &commonpb.MsgBase{
				MsgType:   commonpb.MsgType_kShowPartitions,
				MsgID:     -1, // todo
				Timestamp: 0,  // todo
				SourceID:  -1, // todo
			},
			DbName:         "",
			CollectionName: collectionName,
			CollectionID:   collection.CollectionID,
		})
		if err != nil {
			log.Println(err.Error())
			continue
		}
		err = s.meta.AddCollection(&collectionInfo{
			ID:         collection.CollectionID,
			Schema:     collection.Schema,
			Partitions: partitions.PartitionIDs,
		})
		if err != nil {
			log.Println(err.Error())
			continue
		}
	}
	log.Println("load collection meta from master complete")
	return nil
}

func (s *Server) Stop() error {
	s.ttMsgStream.Close()
	s.msgProducer.Close()
	s.stopServerLoop()
	return nil
}

func (s *Server) stopServerLoop() {
	s.serverLoopCancel()
	s.serverLoopWg.Wait()
}

func (s *Server) GetComponentStates() (*internalpb2.ComponentStates, error) {
	resp := &internalpb2.ComponentStates{
		State: &internalpb2.ComponentInfo{
			NodeID:    Params.NodeID,
			Role:      role,
			StateCode: s.state,
		},
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
		},
	}
	dataNodeStates, err := s.cluster.GetDataNodeStates()
	if err != nil {
		resp.Status.Reason = err.Error()
		return resp, nil
	}
	resp.SubcomponentStates = dataNodeStates
	resp.Status.ErrorCode = commonpb.ErrorCode_SUCCESS
	return resp, nil
}

func (s *Server) GetTimeTickChannel() (*milvuspb.StringResponse, error) {
	return &milvuspb.StringResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_SUCCESS,
		},
		Value: Params.TimeTickChannelName,
	}, nil
}

func (s *Server) GetStatisticsChannel() (*milvuspb.StringResponse, error) {
	return &milvuspb.StringResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_SUCCESS,
		},
		Value: Params.StatisticsChannelName,
	}, nil
}

func (s *Server) RegisterNode(req *datapb.RegisterNodeRequest) (*datapb.RegisterNodeResponse, error) {
	ret := &datapb.RegisterNodeResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
		},
	}
	s.cluster.Register(req.Address.Ip, req.Address.Port, req.Base.SourceID)
	if len(s.ddChannelName) == 0 {
		resp, err := s.masterClient.GetDdChannel()
		if err != nil {
			ret.Status.Reason = err.Error()
			return ret, err
		}
		s.ddChannelName = resp
	}
	ret.Status.ErrorCode = commonpb.ErrorCode_SUCCESS
	ret.InitParams = &internalpb2.InitParams{
		NodeID: Params.NodeID,
		StartParams: []*commonpb.KeyValuePair{
			{Key: "DDChannelName", Value: s.ddChannelName},
			{Key: "SegmentStatisticsChannelName", Value: Params.StatisticsChannelName},
			{Key: "TimeTickChannelName", Value: Params.TimeTickChannelName},
			{Key: "CompleteFlushChannelName", Value: Params.SegmentChannelName},
		},
	}
	return ret, nil
}

func (s *Server) Flush(req *datapb.FlushRequest) (*commonpb.Status, error) {
	success, fails := s.segAllocator.SealAllSegments(req.CollectionID)
	log.Printf("sealing failed segments: %v", fails)
	if !success {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
			Reason:    fmt.Sprintf("flush failed, %d segment can not be sealed", len(fails)),
		}, nil
	}
	return &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_SUCCESS,
	}, nil
}

func (s *Server) AssignSegmentID(req *datapb.AssignSegIDRequest) (*datapb.AssignSegIDResponse, error) {
	resp := &datapb.AssignSegIDResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_SUCCESS,
		},
		SegIDAssignments: make([]*datapb.SegIDAssignment, 0),
	}
	for _, r := range req.SegIDRequests {
		result := &datapb.SegIDAssignment{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
			},
		}
		segmentID, retCount, expireTs, err := s.segAllocator.AllocSegment(r.CollectionID, r.PartitionID, r.ChannelName, int(r.Count))
		if err != nil {
			if _, ok := err.(errRemainInSufficient); !ok {
				result.Status.Reason = fmt.Sprintf("allocation of Collection %d, Partition %d, Channel %s, Count %d error:  %s",
					r.CollectionID, r.PartitionID, r.ChannelName, r.Count, err.Error())
				resp.SegIDAssignments = append(resp.SegIDAssignments, result)
				continue
			}

			log.Printf("no enough space for allocation of Collection %d, Partition %d, Channel %s, Count %d",
				r.CollectionID, r.PartitionID, r.ChannelName, r.Count)
			if err = s.openNewSegment(r.CollectionID, r.PartitionID, r.ChannelName); err != nil {
				result.Status.Reason = fmt.Sprintf("open new segment of Collection %d, Partition %d, Channel %s, Count %d error:  %s",
					r.CollectionID, r.PartitionID, r.ChannelName, r.Count, err.Error())
				resp.SegIDAssignments = append(resp.SegIDAssignments, result)
				continue
			}

			segmentID, retCount, expireTs, err = s.segAllocator.AllocSegment(r.CollectionID, r.PartitionID, r.ChannelName, int(r.Count))
			if err != nil {
				result.Status.Reason = fmt.Sprintf("retry allocation of Collection %d, Partition %d, Channel %s, Count %d error:  %s",
					r.CollectionID, r.PartitionID, r.ChannelName, r.Count, err.Error())
				resp.SegIDAssignments = append(resp.SegIDAssignments, result)
				continue
			}
		}

		result.Status.ErrorCode = commonpb.ErrorCode_SUCCESS
		result.CollectionID = r.CollectionID
		result.SegID = segmentID
		result.PartitionID = r.PartitionID
		result.Count = uint32(retCount)
		result.ExpireTime = expireTs
		result.ChannelName = r.ChannelName
		resp.SegIDAssignments = append(resp.SegIDAssignments, result)
	}
	return resp, nil
}

func (s *Server) openNewSegment(collectionID UniqueID, partitionID UniqueID, channelName string) error {
	group, err := s.insertChannelMgr.GetChannelGroup(collectionID, channelName)
	if err != nil {
		return err
	}
	segmentInfo, err := s.meta.BuildSegment(collectionID, partitionID, group)
	if err != nil {
		return err
	}
	if err = s.meta.AddSegment(segmentInfo); err != nil {
		return err
	}
	if err = s.segAllocator.OpenSegment(segmentInfo); err != nil {
		return err
	}
	return nil
}

func (s *Server) ShowSegments(req *datapb.ShowSegmentRequest) (*datapb.ShowSegmentResponse, error) {
	ids := s.meta.GetSegmentsByCollectionAndPartitionID(req.CollectionID, req.PartitionID)
	return &datapb.ShowSegmentResponse{SegmentIDs: ids}, nil
}

func (s *Server) GetSegmentStates(req *datapb.SegmentStatesRequest) (*datapb.SegmentStatesResponse, error) {
	resp := &datapb.SegmentStatesResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
		},
	}

	segmentInfo, err := s.meta.GetSegment(req.SegmentID)
	if err != nil {
		resp.Status.Reason = "get segment states error: " + err.Error()
		return resp, nil
	}
	resp.State = segmentInfo.State
	resp.CreateTime = segmentInfo.OpenTime
	resp.SealedTime = segmentInfo.SealedTime
	resp.FlushedTime = segmentInfo.FlushedTime
	// TODO start/end positions
	return resp, nil
}

func (s *Server) GetInsertBinlogPaths(req *datapb.InsertBinlogPathRequest) (*datapb.InsertBinlogPathsResponse, error) {
	panic("implement me")
}

func (s *Server) GetInsertChannels(req *datapb.InsertChannelRequest) (*internalpb2.StringList, error) {
	resp := &internalpb2.StringList{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_SUCCESS,
		},
	}
	contains, ret := s.insertChannelMgr.ContainsCollection(req.CollectionID)
	if contains {
		resp.Values = ret
		return resp, nil
	}
	channelGroups, err := s.insertChannelMgr.AllocChannels(req.CollectionID, s.cluster.GetNumOfNodes())
	if err != nil {
		resp.Status.ErrorCode = commonpb.ErrorCode_UNEXPECTED_ERROR
		resp.Status.Reason = err.Error()
		return resp, nil
	}

	channels := make([]string, Params.InsertChannelNumPerCollection)
	for _, group := range channelGroups {
		channels = append(channels, group...)
	}
	s.cluster.WatchInsertChannels(channelGroups)

	resp.Values = channels
	return resp, nil
}

func (s *Server) GetCollectionStatistics(req *datapb.CollectionStatsRequest) (*datapb.CollectionStatsResponse, error) {
	// todo implement
	return nil, nil
}

func (s *Server) GetPartitionStatistics(req *datapb.PartitionStatsRequest) (*datapb.PartitionStatsResponse, error) {
	// todo implement
	return nil, nil
}
